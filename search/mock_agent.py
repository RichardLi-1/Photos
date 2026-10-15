"""Mock photo agent: a rule-based Planner (no LLM) + an Executor that owns confirmation and undo. Local only.

THE SEAM -- plugging in your own agent:
  1. Write a class with `plan(message, ctx) -> Plan` (the Planner protocol below), e.g. an LLM tool-calling loop over
     tools.schemas(): let the model call read-only tools through `ctx.read(name, args)`, and return its mutating calls
     as `Plan(steps=[Step(tool, args, description)], reply=...)`.
  2. Use it: `Executor(planner=MyPlanner())`, or set `api_agent.EXECUTOR.planner = MyPlanner()`. Nothing else changes:
     the Executor still dry-runs every DESTRUCTIVE step, writes the confirmation summary from the dry run (not the model's
     text), replays with the confirm_token only after /confirm, and records an undo for every mutation it ran.

Run a phrase from the shell:  .venv/bin/python mock_agent.py "clean up duplicates"   (preview only; nothing changes)
"""
from __future__ import annotations

import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from photosearch import tools

ACTOR = "mock-agent"
SESSION_TTL_S = 30 * 60
MAX_SESSIONS = 200
MAX_TEXT = 500
MAX_UNDO = 20
ALBUM_N = 30        # photos in "make an album of my X photos"
FIND_N = 60         # photos for "find X"
TRASH_QUERY_N = 24  # photos for "delete photos of X" (always previewed first)

DESTRUCTIVE = frozenset(tools.DESTRUCTIVE)
READ_ONLY = frozenset(tools.REGISTRY) - tools.MUTATING
CHUNK = {"trash_photos": tools.MAX_TRASH, "restore_photos": tools.MAX_TRASH,
         "add_to_album": tools.MAX_ALBUM_BATCH, "remove_from_album": tools.MAX_ALBUM_BATCH}


# ================================================================ the seam: plan types + Planner protocol

class NothingToDo(Exception):
    """Raise from a step's args function or a reply function: the plan stops there and str(e) becomes the reply."""


@dataclass
class Step:
    tool: str          # a name from tools.REGISTRY
    args: Any          # dict, or fn(outs) -> dict, where outs[i] is the output of plan step i (may raise NothingToDo)
    description: str   # one line for the step checklist ("Looked for exact duplicates")


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    reply: Any = ""            # str, or fn(outs) -> str, evaluated after the steps ran (destructive ones as dry runs)
    intent: str = "unknown"
    undo_last: bool = False    # "undo": the Executor runs this session's latest undo instead of steps


@dataclass
class Context:
    session_id: str
    history: list[dict]                     # [{"user": ..., "reply": ...}], oldest first
    last_photo_ids: list[int]               # what the wall showed after the previous turn ("delete these")
    read: Callable[..., dict]               # read(tool, args, description=None): read-only tools only, logged as a step


class Planner(Protocol):
    def plan(self, message: str, ctx: Context) -> Plan: ...


# ================================================================ RulePlanner: keyword/regex intents, no model

KIND_LABEL = {"exact_duplicate": "exact duplicates", "near_duplicate": "near-duplicates", "blurry": "blurry photos",
              "dark": "dark photos", "overexposed": "overexposed photos", "screenshot": "screenshots",
              "receipt": "receipts", "document": "documents", "whiteboard": "whiteboard photos", "meme": "memes",
              "accidental": "accidental shots"}
KIND_WORDS = [  # order matters: "near-duplicates" before "duplicates"
    ("near_duplicate", r"near[- ]?dup\w*|similar|bursts?|look-?alikes?|almost (?:the )?same"),
    ("exact_duplicate", r"exact\w*|identical|dup(?:e|es|licate|licates)?\b|copies"),
    ("blurry", r"blur\w*|out of focus|shaky"),
    ("overexposed", r"over-?exposed|too bright|washed[- ]out|blown[- ]out"),
    ("dark", r"dark|under-?exposed"),
    ("screenshot", r"screen ?shots?|screen ?caps?"),
    ("receipt", r"receipts?"),
    ("document", r"documents?|paperwork"),
    ("whiteboard", r"white ?boards?"),
    ("meme", r"memes?"),
    ("accidental", r"accident\w*|pocket shots?|by mistake"),
]
CAUTION = {"receipt", "document", "whiteboard"}  # may still matter: say so
PREFER = ["exact_duplicate", "near_duplicate", "blurry", "accidental", "dark", "overexposed", "screenshot", "meme",
          "whiteboard", "receipt", "document"]

PRONOUN = re.compile(r"^(?:all )?(?:of )?(?:these|those|them|this|it|this one|that|the results|the ones shown|"
                     r"(?:these|those|the) (?:photos|pictures|pics|results|ones))$", re.I)
VERB_DELETE = re.compile(r"\b(delete|trash|remove|get rid of|bin|throw (?:away|out)|toss|discard|erase)\b", re.I)
VERB_CLEAN = re.compile(r"\b(clean ?up|clean|tidy(?: up)?|declutter|free up|sort out)\b", re.I)
LIBRARY = r"^(?:my |the )?(?:library|photos|camera roll|phone|mac)$"
NOT_NEEDED = re.compile(r"(?:don'?t|do not|won'?t) (?:really )?need|unneeded|unnecessary|\bjunk\b|clutter", re.I)

HELP = ("I can make albums (“make an album of my beach photos”, “album: Japan trip”), add photos to them "
        "(“add these to Japan trip”), clean up duplicates, near-duplicates, blurry or dark shots, screenshots, "
        "whiteboards and receipts, move photos to Recently Deleted (“delete these”) and bring them back (“restore”, "
        "“undo”), and show things (“show my albums”, “show trash”, “find dogs in snow”). Anything that removes photos "
        "waits for your OK, and everything can be undone.")
CANT = ("I can’t do that yet. I can make albums, clean up duplicates or blurry shots, move photos to Recently Deleted "
        "and bring them back. Try “clean up duplicates” or “what can you do”.")


def _plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


def clean_query(s: str) -> str:
    """'all of my beach photos' -> 'beach'; 'photos of receipts' -> 'receipts'."""
    s = s.strip().strip(".!?\"'“”‘’ ")
    s = re.sub(r"^(?:all\s+)?(?:of\s+)?(?:my|the|our|some|any)\s+", "", s, flags=re.I)
    s = re.sub(r"^(?:photos|pictures|pics|images|shots|snaps)\s+(?:of|from|with|showing|at|in)\s+(?:my\s+|the\s+)?", "", s, flags=re.I)
    s = re.sub(r"\s+(?:photos|pictures|pics|images|shots|snaps|ones)$", "", s, flags=re.I)
    s = re.sub(r"^(?:my|the)\s+", "", s, flags=re.I)
    return s.strip()


def kind_of(text: str) -> str | None:
    for k, pat in KIND_WORDS:
        if re.search(rf"\b(?:{pat})", text, re.I):
            return k
    return None


def _title(q: str) -> str:
    return q[:1].upper() + q[1:] if q else q


def _find_album(albums: list[dict], name: str) -> dict | None:
    n = name.strip().strip("\"'“”").lower()
    for test in (lambda a: a["name"].lower() == n, lambda a: a["name"].lower().startswith(n),
                 lambda a: n in a["name"].lower()):
        hit = [a for a in albums if test(a)]
        if hit:
            return hit[0]
    return None


def _album_names(albums):
    return ", ".join(f"“{a['name']}”" for a in albums[:8]) or "none yet"


def _ids_of(out: dict) -> list[int]:
    return [r["id"] for r in out.get("results") or out.get("photos") or []]


class RulePlanner:
    """Keyword/regex intents. Each method returns a Plan; nothing here touches the library except through ctx.read."""

    def plan(self, message: str, ctx: Context) -> Plan:
        t = re.sub(r"\s+", " ", message).strip().rstrip(".!")
        low = t.lower()
        if not low or re.search(r"^(?:help|\?|commands)$|what can you do|what do you do|how does this work|what can i say", low):
            return Plan(reply=HELP, intent="help")
        if re.match(r"^(?:undo|undo (?:that|it|this)|revert(?: that)?|never ?mind|put (?:it|them) back|take that back)$", low):
            return Plan(undo_last=True, intent="undo")
        if re.search(r"\b(?:empty|purge|clear)\b.*\b(?:trash|recently deleted|bin)\b|permanently delete", low):
            return Plan(reply="Emptying Recently Deleted can’t be undone, so I leave that one to you. Photos there stay "
                              "restorable, and files on disk are never deleted.", intent="cant")
        if re.search(r"\b(?:restore|undelete|recover|bring (?:\w+ )?back|un-?trash)\b", low):
            return self.restore(t, ctx)
        if m := re.match(r"^rename (?:the |my )?(?:album )?[\"“']?(.+?)[\"”']?(?: album)? (?:to|as) [\"“']?(.+?)[\"”']?$", t, re.I):
            return self.rename_album(m[1], m[2], ctx)
        if m := (re.match(r"^(?:delete|remove|trash) (?:the |my )?album[: ]+[\"“']?(.+?)[\"”']?$", t, re.I)
                 or re.match(r"^(?:delete|remove|trash) (?:the |my )?[\"“']?(.+?)[\"”']? album$", t, re.I)):
            return self.delete_album(m[1], ctx)
        m = re.match(r"^(?:remove|take) (.+?) (?:out of|from) (?:the |my )?(?:album )?[\"“']?(.+?)[\"”']?(?: album)?$", t, re.I)
        if m and not re.match(LIBRARY, m[2], re.I):
            return self.remove_from_album(m[1], m[2], ctx)
        if re.match(r"^(?:(?:show|list|open|see|view|browse)(?: me)?(?: all)?(?: (?:of )?(?:my|the))? albums|(?:my )?albums)$", low):
            return self.show_albums()
        if re.match(r"^(?:(?:show|open|view|see|list|what'?s in)(?: me)?(?: (?:my|the))? (?:trash|recently deleted|bin|deleted(?: photos)?)|trash|recently deleted)$", low):
            return self.show_trash()
        if m := (re.match(r"^(?:show|open|view|see)(?: me)?(?: (?:my|the))? album[: ]+[\"“']?(.+?)[\"”']?$", t, re.I)
                 or re.match(r"^(?:show|open|view)(?: me)?(?: (?:my|the))? [\"“']?(.+?)[\"”']? album$", t, re.I)):
            return self.show_album(m[1], ctx)
        if m := re.match(r"^(?:put|add|save|move|copy) (.+?) (?:in|into|to) (?:a )?new album(?: (?:called|named) [\"“']?(.+?)[\"”']?)?$", t, re.I):
            return self.create_album(m[2] or None, m[1], ctx)
        if re.search(r"\b(?:make|create|start|build|new)\b.*\balbum\b", low) or re.match(r"^album\s*:", low):
            name, src = self._album_request(t)
            return self.create_album(name, src, ctx)
        if m := re.match(r"^(?:add|put|save|move|copy) (.+?) (?:to|into|in) (?:the |my )?(?:album )?[\"“']?(.+?)[\"”']?(?: album)?$", t, re.I):
            if re.match(r"^(?:the )?(?:trash|bin|recently deleted)$", m[2], re.I):
                return self.trash_last(ctx) if PRONOUN.match(m[1]) else self.trash_query(clean_query(m[1]))
            return self.add_to_album(m[1], m[2], ctx)
        kind = kind_of(low)
        deleting, cleaning, unneeded = VERB_DELETE.search(low), VERB_CLEAN.search(low), NOT_NEEDED.search(low)
        if kind and (deleting or cleaning or unneeded or re.match(r"^(?:find|show)", low)):
            return self.cleanup(kind, ctx, review=not (deleting or cleaning or unneeded))
        if cleaning or unneeded:
            return self.cleanup(None, ctx)
        if deleting:
            target = re.sub(r"^.*?\b(?:delete|trash|remove|get rid of|bin|throw (?:away|out)|toss|discard|erase)\b\s*", "", t, flags=re.I)
            target = re.sub(r"^(?:all\s+)?(?:of\s+)?", "", target, flags=re.I)
            target = re.sub(r"\s+(?:from|in|out of) (?:my |the )?(?:library|photos|camera roll|phone|mac)$", "", target, flags=re.I)
            if not target or PRONOUN.match(target):
                return self.trash_last(ctx)
            return self.trash_query(clean_query(target))
        if m := re.match(r"^(?:show|find|search|look for|get|display)(?: me)?(?: for)? (.+)$", t, re.I):
            q = clean_query(m[1])
            if q:
                return self.find(q)
        return Plan(reply=CANT, intent="cant")

    # ---------------------------------------------------------------- albums

    @staticmethod
    def _album_request(t: str) -> tuple[str | None, str | None]:
        """-> (album name or None, photo source phrase or None)."""
        if m := re.match(r"^\s*album\s*:\s*(.+)$", t, re.I):
            return m[1].strip(), None
        i = re.search(r"\balbum\b", t, re.I)
        before, after = t[:i.start()], t[i.end():]
        name = None
        if m := re.search(r"\b(?:called|named|titled)\s+[\"“']?(.+?)[\"”']?(?=\s+(?:with|of|from|using|for|containing)\b|$)", after, re.I):
            name, after = m[1], after[:m.start()] + after[m.end():]
        elif m := re.search(r"[\"“']([^\"”']+)[\"”']", t):
            name = m[1]
            before, after = before.replace(m[0], ""), after.replace(m[0], "")
        src = None
        if m := re.search(r"\b(?:of|with|from|for|using|containing)\s+(.+)$", after, re.I):
            src = m[1]
        else:
            adj = re.sub(r"^.*?\b(?:make|create|start|build)\b\s*(?:me\s+|us\s+)?", "", before, flags=re.I)
            adj = re.sub(r"^(?:an?|the|my)\s+", "", adj.strip(), flags=re.I)
            adj = re.sub(r"^new\s+", "", adj, flags=re.I).strip()
            src = adj or None
        return name, src

    def _source(self, src: str | None, name: str | None, ctx: Context):
        """-> ('last', ids) | ('query', q)."""
        if src and PRONOUN.match(src.strip()):
            return "last", ctx.last_photo_ids
        q = clean_query(src) if src else clean_query(name or "")
        return "query", q

    def create_album(self, name: str | None, src: str | None, ctx: Context) -> Plan:
        how, what = self._source(src, name, ctx)
        if how == "last" and not what:
            return Plan(reply="Which photos? Search for some first, then say “make an album of these”.", intent="create_album")
        if how == "query" and not what:
            return Plan(reply="What should go in it? Try “make an album of my beach photos”.", intent="create_album")
        name = (name or _title(what if how == "query" else "New album")).strip()[:120]
        existing = _find_album(ctx.read("list_albums", {}, "Checked your albums")["albums"], name)
        if existing and existing["name"].lower() == name.lower():
            return self.add_to_album(src or name, existing["name"], ctx)
        steps = []
        if how == "query":
            steps.append(Step("search_photos", {"query": what, "limit": ALBUM_N}, f"Found photos of “{what}”"))
            ids = lambda outs: {"name": name, "photo_ids": _need(_ids_of(outs[0]), f"I couldn’t find photos of “{what}”.")}
        else:
            ids = {"name": name, "photo_ids": list(what)[:tools.MAX_ALBUM_BATCH]}
        steps.append(Step("create_album", ids, f"Made the album “{name}”"))
        k = len(steps) - 1
        steps.append(Step("get_album", lambda outs: {"album_id": outs[k]["album"]["id"], "limit": 200}, "Opened it"))
        how_txt = f" that match “{what}”" if how == "query" else ""
        return Plan(steps, lambda outs: f"Made the album “{name}” with {_plural(outs[k]['added'], 'photo')}{how_txt}.",
                    intent="create_album")

    def add_to_album(self, src: str, album_name: str, ctx: Context) -> Plan:
        albums = ctx.read("list_albums", {}, "Checked your albums")["albums"]
        a = _find_album(albums, album_name)
        if a is None:
            return self.create_album(_title(album_name.strip().strip("\"'“”")), src, ctx)
        how, what = self._source(src, None, ctx)
        if not what:
            return Plan(reply="Which photos? Search for some first, then say “add these to …”.", intent="add_to_album")
        steps = []
        if how == "query":
            steps.append(Step("search_photos", {"query": what, "limit": ALBUM_N}, f"Found photos of “{what}”"))
            args = lambda outs: {"album_id": a["id"], "photo_ids": _need(_ids_of(outs[0]), f"I couldn’t find photos of “{what}”.")}
        else:
            args = {"album_id": a["id"], "photo_ids": list(what)[:tools.MAX_ALBUM_BATCH]}
        steps.append(Step("add_to_album", args, f"Added them to “{a['name']}”"))
        k = len(steps) - 1
        steps.append(Step("get_album", {"album_id": a["id"], "limit": 200}, "Opened the album"))

        def reply(outs):
            o = outs[k]
            extra = f" ({o['already_in_album']} were already there)" if o.get("already_in_album") else ""
            return f"Added {_plural(o['added'], 'photo')} to “{a['name']}”{extra}."
        return Plan(steps, reply, intent="add_to_album")

    def remove_from_album(self, src: str, album_name: str, ctx: Context) -> Plan:
        a = _find_album(ctx.read("list_albums", {}, "Checked your albums")["albums"], album_name)
        if a is None:
            return Plan(reply=f"I couldn’t find an album called “{album_name}”.", intent="remove_from_album")
        how, what = self._source(src, None, ctx)
        steps = []
        if how == "query":
            if not what:
                return Plan(reply="Which photos should come out?", intent="remove_from_album")
            steps.append(Step("search_photos", {"query": what, "album_id": a["id"], "limit": 60},
                              f"Found “{what}” in “{a['name']}”"))
            args = lambda outs: {"album_id": a["id"], "photo_ids": _need(_ids_of(outs[0]), f"No photos of “{what}” in “{a['name']}”.")}
        else:
            if not what:
                return Plan(reply="Which photos? Open the album first, then say “remove these from …”.", intent="remove_from_album")
            args = {"album_id": a["id"], "photo_ids": list(what)}
        steps.append(Step("remove_from_album", args, f"Take them out of “{a['name']}”"))
        return Plan(steps, lambda outs: f"Take these out of “{a['name']}”? They stay in your library.", intent="remove_from_album")

    def delete_album(self, name: str, ctx: Context) -> Plan:
        albums = ctx.read("list_albums", {}, "Checked your albums")["albums"]
        a = _find_album(albums, name)
        if a is None:
            return Plan(reply=f"I couldn’t find an album called “{name}”. Your albums: {_album_names(albums)}.", intent="delete_album")
        return Plan([Step("get_album", {"album_id": a["id"], "limit": 200}, f"Opened “{a['name']}”"),
                     Step("delete_album", {"album_id": a["id"]}, f"Delete the album “{a['name']}”")],
                    f"Delete the album “{a['name']}”? Its {_plural(a['count'], 'photo')} stay in your library.",
                    intent="delete_album")

    def rename_album(self, old: str, new: str, ctx: Context) -> Plan:
        albums = ctx.read("list_albums", {}, "Checked your albums")["albums"]
        a = _find_album(albums, old)
        if a is None:
            return Plan(reply=f"I couldn’t find an album called “{old}”. Your albums: {_album_names(albums)}.", intent="rename_album")
        return Plan([Step("rename_album", {"album_id": a["id"], "new_name": new}, f"Renamed “{a['name']}” to “{new}”")],
                    f"Renamed “{a['name']}” to “{new}”.", intent="rename_album")

    def show_albums(self) -> Plan:
        def reply(outs):
            al = outs[0]["albums"]
            if not al:
                return "You don’t have any albums yet. Try “make an album of my beach photos”."
            return f"You have {_plural(len(al), 'album')}: " + ", ".join(f"{a['name']} ({a['count']})" for a in al[:12]) + "."
        return Plan([Step("list_albums", {}, "Listed your albums")], reply, intent="show_albums")

    def show_album(self, name: str, ctx: Context) -> Plan:
        albums = ctx.read("list_albums", {}, "Checked your albums")["albums"]
        a = _find_album(albums, name)
        if a is None:
            return Plan(reply=f"I couldn’t find an album called “{name}”. Your albums: {_album_names(albums)}.", intent="show_album")
        return Plan([Step("get_album", {"album_id": a["id"], "limit": 200}, f"Opened “{a['name']}”")],
                    f"“{a['name']}”: {_plural(a['count'], 'photo')}.", intent="show_album")

    def show_trash(self) -> Plan:
        def reply(outs):
            n = outs[0]["total"]
            return "Recently Deleted is empty." if not n else \
                f"{_plural(n, 'photo')} in Recently Deleted. Say “restore” to bring them back."
        return Plan([Step("list_trash", {"limit": 200}, "Opened Recently Deleted")], reply, intent="show_trash")

    def find(self, q: str) -> Plan:
        return Plan([Step("search_photos", {"query": q, "limit": FIND_N}, f"Searched for “{q}”")],
                    lambda outs: f"Here are the {outs[0]['count']} best matches for “{q}”." if outs[0]["count"]
                    else f"Nothing matched “{q}”.", intent="find")

    # ---------------------------------------------------------------- trash / restore / cleanup

    def restore(self, t: str, ctx: Context) -> Plan:
        rest = re.sub(r"^.*?\b(?:restore|undelete|recover|bring|un-?trash)\b\s*", "", t, flags=re.I)
        rest = re.sub(r"\s*\bback\b\s*", " ", rest).strip()
        if rest and PRONOUN.match(rest) and ctx.last_photo_ids:
            return Plan([Step("restore_photos", {"photo_ids": ctx.last_photo_ids[:tools.MAX_TRASH]}, "Restored them")],
                        lambda outs: _restored(outs[0]), intent="restore")
        return Plan([Step("list_trash", {"limit": tools.MAX_TRASH}, "Opened Recently Deleted"),
                     Step("restore_photos", lambda outs: {"photo_ids": _need(_ids_of(outs[0]), "Recently Deleted is empty, so there’s nothing to restore.")},
                          "Restored them")],
                    lambda outs: _restored(outs[1]), intent="restore")

    def trash_last(self, ctx: Context) -> Plan:
        ids = ctx.last_photo_ids[:tools.MAX_TRASH]
        if not ids:
            return Plan(reply="Which photos? Search for them first, then say “delete these”.", intent="trash")
        return Plan([Step("trash_photos", {"photo_ids": ids, "reason": "deleted by you via the agent"},
                          f"Move {_plural(len(ids), 'photo')} to Recently Deleted")],
                    lambda outs: f"Move these {_plural(outs[0].get('would_trash', 0), 'photo')} to Recently Deleted? "
                                 "You can restore them any time.", intent="trash")

    def trash_query(self, q: str) -> Plan:
        if not q:
            return Plan(reply="Which photos? Try “delete photos of receipts” or search first and say “delete these”.", intent="trash")
        return Plan([Step("search_photos", {"query": q, "limit": TRASH_QUERY_N}, f"Searched for “{q}”"),
                     Step("trash_photos", lambda outs: {"photo_ids": _need(_ids_of(outs[0]), f"Nothing matched “{q}”."),
                                                        "reason": f"matched “{q[:120]}”"},
                          "Move them to Recently Deleted")],
                    lambda outs: f"These are the {_plural(outs[1].get('would_trash', 0), 'photo')} that best match “{q}”. "
                                 "Check them before you confirm: they’ll go to Recently Deleted.", intent="trash")

    def cleanup(self, kind: str | None, ctx: Context, review: bool = False) -> Plan:
        others = ""
        if kind is None:  # "clean up" / "photos I probably don't need": safest non-empty kind first, mention the rest
            summary = ctx.read("find_cleanup_candidates", {"limit": 1}, "Checked for photos you may not need")["summary"]
            kind = next((k for k in PREFER if summary.get(k)), None)
            if kind is None:
                return Plan(reply="Your library looks tidy: no duplicates, blurry shots or screenshots found.", intent="cleanup")
            rest = [_count_label(k, summary[k]) for k in PREFER if k != kind and summary.get(k)]
            if rest:
                others = " I also found " + ", ".join(rest[:4]) + "; ask for any of them next."
        label = KIND_LABEL[kind]
        grouped = kind in ("exact_duplicate", "near_duplicate")
        if review:  # "show me blurry photos": just look, no pending action
            def look(outs):
                n = outs[0]["total"]
                if not n:
                    raise NothingToDo(_none_found(kind, outs[0]["summary"]))
                return (f"Here are {_count_label(kind, n)}. Say “clean up {label}” to move the "
                        f"{'extra copies' if grouped else 'ones you don’t need'} to Recently Deleted.")
            return Plan([Step("find_cleanup_candidates", {"kind": kind, "limit": 200}, f"Looked for {label}")], look,
                        intent="review")

        def ids(outs):
            c = outs[0]["candidates"]
            out = [p for x in c for p in (x["remove"] if "remove" in x else x["photo_ids"])]
            if not out:
                raise NothingToDo(_none_found(kind, outs[0]["summary"]))
            return {"photo_ids": out[:tools.MAX_TRASH * 5], "reason": f"cleanup: {label[:-1] if label.endswith('s') else label}"}

        def reply(outs):
            n_found, n = outs[0]["total"], outs[1].get("would_trash", 0)
            if grouped:
                msg = (f"I found {_plural(n_found, 'group')} of {label}. I’ll keep the best photo of each and move the "
                       f"{_plural(n, 'extra copy', 'extra copies')} to Recently Deleted.")
            else:
                msg = f"I found {_count_label(kind, n)}. Move {'it' if n == 1 else 'them'} to Recently Deleted?"
            if kind in CAUTION:
                msg += " These can still be important, so check them first."
            return msg + " Nothing is deleted from disk, and you can undo." + others
        return Plan([Step("find_cleanup_candidates", {"kind": kind, "limit": 200}, f"Looked for {label}"),
                     Step("trash_photos", ids, f"Move the {'extra copies' if grouped else label} to Recently Deleted")],
                    reply, intent="cleanup")


def _count_label(kind: str, n: int) -> str:
    """'124 groups of near-duplicates', '1 overexposed photo', '5 blurry photos'."""
    label = KIND_LABEL[kind]
    if kind.endswith("duplicate"):
        return f"{_plural(n, 'group')} of {label}"
    return f"{n} {label[:-1] if n == 1 and label.endswith('s') else label}"


def _need(ids: list[int], msg: str) -> list[int]:
    if not ids:
        raise NothingToDo(msg)
    return ids


def _restored(o: dict) -> str:
    n = o.get("restored", 0)
    return f"Restored {_plural(n, 'photo')} from Recently Deleted." if n else "Nothing to restore: those photos aren’t in Recently Deleted."


def _none_found(kind: str, summary: dict) -> str:
    msg = f"I didn’t find any {KIND_LABEL[kind]} in your library."
    alt = "near_duplicate" if kind != "near_duplicate" and summary.get("near_duplicate") else \
        next((k for k in PREFER if k != kind and summary.get(k)), None)
    if alt:
        msg += f" I did find {_count_label(alt, summary[alt])}, though: try “clean up {KIND_LABEL[alt]}”."
    return msg


# ================================================================ Executor: runs plans, owns confirmation + undo

class AgentError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class _Call:
    tool: str
    args: dict
    token: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class _Pending:
    id: str
    created: float
    calls: list[_Call]
    steps: list[dict]
    summary: str
    count: int
    photo_ids: list[int]
    groups: list[dict] | None


@dataclass
class _Undo:
    id: str
    label: str
    calls: list[_Call]   # run in order (already reversed)


@dataclass
class _Session:
    id: str
    touched: float
    history: list[dict] = field(default_factory=list)
    last_photo_ids: list[int] = field(default_factory=list)
    pending: dict = field(default_factory=dict)
    undos: OrderedDict = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(9)}"


class Executor:
    """Runs a Planner's plan through tools.call with the safety policy:

    * read-only steps run immediately; non-destructive mutations (create/add/rename/restore) run immediately with an undo;
    * DESTRUCTIVE steps (trash, remove-from-album, delete album, purge) run as dry runs only and become a pending
      confirmation whose summary comes from the dry run; /confirm replays them with their confirm_tokens;
    * every mutation records an undo (restore_photos for trash, delete_album / remove_from_album for album changes, ...),
      whose own destructive calls are confirmed automatically because the user asked for the undo;
    * once a plan hits a destructive step, every later step must be destructive too (they are all previewed together).
    """

    def __init__(self, planner: Planner | None = None, call: Callable[..., dict] = tools.call, ttl_s: float = SESSION_TTL_S,
                 clock: Callable[[], float] = time.time):
        self.planner = planner or RulePlanner()
        self._call, self.ttl, self.clock = call, ttl_s, clock
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._mu = threading.Lock()

    # ---------------------------------------------------------------- sessions

    def _session(self, sid: str | None, create: bool) -> _Session:
        now = self.clock()
        with self._mu:
            for k in [k for k, s in self._sessions.items() if now - s.touched > self.ttl]:
                del self._sessions[k]
            s = self._sessions.get(sid) if sid else None
            if s is None:
                if not create:
                    raise AgentError(404, "unknown or expired session; send a new message")
                s = _Session(_new_id("s"), now)
                self._sessions[s.id] = s
                while len(self._sessions) > MAX_SESSIONS:
                    self._sessions.popitem(last=False)
            s.touched = now
            self._sessions.move_to_end(s.id)
            return s

    def call(self, tool: str, args: dict) -> dict:
        return self._call(tool, args, actor=ACTOR)

    # ---------------------------------------------------------------- public API

    def message(self, text: Any, session_id: str | None = None) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise AgentError(400, "text required")
        if len(text) > MAX_TEXT:
            raise AgentError(400, f"text longer than {MAX_TEXT} characters")
        s = self._session(session_id if isinstance(session_id, str) else None, create=True)
        with s.lock:
            out = self._message(s, text.strip())
            s.history = (s.history + [{"user": text.strip(), "reply": out["reply"]}])[-20:]
            if out.get("results", {}).get("photo_ids"):
                s.last_photo_ids = out["results"]["photo_ids"]
            return out

    def confirm(self, session_id: str, pending_id: str) -> dict:
        s = self._session(session_id, create=False)
        with s.lock:
            p = self._take_pending(s, pending_id)
            steps = [dict(x) for x in p.steps]
            undo_calls, done_outs, error = [], [], None
            for c in p.calls:
                o = self.call(c.tool, {**c.args, "dry_run": False, "confirm_token": c.token})
                if not o.get("ok"):
                    error = o.get("error", "error")
                    break
                done_outs.append((c, o))
                undo_calls += self._undo_for(c, o)
            for x in steps:
                if x["status"] == "pending":
                    x["status"] = "done" if error is None else ("failed" if not done_outs else "partial")
            reply = "; ".join(t for t in (_done_text(c.tool, o) for c, o in _group(done_outs)) if t)
            reply = (f"Done: {reply}." if reply else "") if error is None else \
                (f"Done in part: {reply}. " if reply else "") + f"Then this failed: {error}"
            affected = [i for c, o in done_outs for i in (o.get("photo_ids") or [])]
            resp = {"session_id": s.id, "reply": reply.strip(), "steps": steps,
                    "results": {"photo_ids": affected, "view": "affected"} if affected else None}
            u = self._remember_undo(s, undo_calls)
            if u:
                resp["undo"] = u
            if any(c.tool == "purge_trash" for c, _ in done_outs):
                resp["reply"] += " This one can’t be undone."
            s.history.append({"user": "(confirmed)", "reply": resp["reply"]})
            return _clean(resp)

    def cancel(self, session_id: str, pending_id: str) -> dict:
        s = self._session(session_id, create=False)
        with s.lock:
            p = self._take_pending(s, pending_id, allow_expired=True)
            steps = [dict(x, status="cancelled") if x["status"] == "pending" else dict(x) for x in p.steps]
            return {"session_id": s.id, "reply": "Cancelled. Nothing was changed.", "steps": steps}

    def undo(self, session_id: str, undo_id: str | None = None) -> dict:
        s = self._session(session_id, create=False)
        with s.lock:
            return self._run_undo(s, undo_id)

    # ---------------------------------------------------------------- internals

    def _take_pending(self, s: _Session, pid: str, allow_expired: bool = False) -> _Pending:
        p = s.pending.pop(pid, None) if isinstance(pid, str) else None
        if p is None:
            raise AgentError(404, "nothing waiting for confirmation with that id (already confirmed or cancelled?)")
        if not allow_expired and self.clock() - p.created > tools.PREVIEW_TTL_S:
            raise AgentError(410, "this preview expired; ask again to get a fresh one")
        return p

    def _run_undo(self, s: _Session, undo_id: str | None) -> dict:
        if undo_id is None:
            if not s.undos:
                return {"session_id": s.id, "reply": "There’s nothing to undo yet.", "steps": []}
            undo_id = next(reversed(s.undos))
        u = s.undos.pop(undo_id, None) if isinstance(undo_id, str) else None
        if u is None:
            raise AgentError(404, "nothing to undo with that id (already undone?)")
        steps, outs, error = [], [], None
        for c in u.calls:
            o = self._run_auto_confirmed(c)
            ok = bool(o.get("ok"))
            steps.append({"tool": c.tool, "description": _undo_label([c]), "status": "done" if ok else "failed"})
            if not ok:
                error = o.get("error", "error")
                break
            outs.append((c, o))
        txt = "; ".join(t for t in (_done_text(c.tool, o) for c, o in _group(outs)) if t)
        reply = f"Undone: {txt}." if error is None else f"Undo stopped: {error}" + (f" (done so far: {txt})" if txt else "")
        affected = [i for c, o in outs for i in (o.get("photo_ids") or [])]
        resp = {"session_id": s.id, "reply": reply, "steps": steps,
                "results": {"photo_ids": affected, "view": "affected"} if affected else None}
        s.history.append({"user": "(undo)", "reply": reply})
        return _clean(resp)

    def _run_auto_confirmed(self, c: _Call) -> dict:
        """Undo calls are user-initiated: destructive ones still go dry run -> token -> real run, confirmed here."""
        if c.tool not in DESTRUCTIVE:
            return self.call(c.tool, dict(c.args))
        pre = self.call(c.tool, {**c.args, "dry_run": True})
        if not pre.get("ok"):
            return pre
        if not pre.get("confirm_token"):
            return {**pre, "ok": True, "photo_ids": []}  # nothing left to undo
        return self.call(c.tool, {**c.args, "dry_run": False, "confirm_token": pre["confirm_token"]})

    def _message(self, s: _Session, text: str) -> dict:
        steps: list[dict] = []
        reads: list[tuple[str, dict]] = []

        def read(tool: str, args: dict | None = None, description: str | None = None) -> dict:
            if tool not in READ_ONLY:
                raise AgentError(400, f"ctx.read only runs read-only tools; {tool!r} must be a plan step")
            o = self.call(tool, dict(args or {}))
            steps.append({"tool": tool, "description": description or tool.replace("_", " "),
                          "status": "done" if o.get("ok") else "failed"})
            if not o.get("ok"):
                raise NothingToDo(f"Something went wrong: {o.get('error')}")
            reads.append((tool, o))
            return o

        ctx = Context(s.id, list(s.history), list(s.last_photo_ids), read)
        try:
            plan = self.planner.plan(text, ctx)
        except NothingToDo as e:
            return _clean({"session_id": s.id, "reply": str(e), "steps": steps, "results": _results(reads)})
        if plan.undo_last:
            return self._run_undo(s, None)

        outs: list[dict] = []
        ran: list[tuple[str, dict]] = list(reads)
        pending_calls: list[_Call] = []
        pending_outs: list[tuple[_Call, dict]] = []
        undo_calls: list[_Call] = []
        reply, stopped = None, False
        for i, st in enumerate(plan.steps):
            entry = {"tool": st.tool, "description": st.description, "status": "skipped"}
            steps.append(entry)
            if stopped:
                continue
            if pending_outs and st.tool not in DESTRUCTIVE:
                entry["status"] = "failed"
                reply, stopped = "Plan error: only destructive steps may follow a destructive one.", True
                continue
            if st.tool not in tools.REGISTRY:
                entry["status"] = "failed"
                reply, stopped = f"Plan error: unknown tool {st.tool!r}.", True
                continue
            try:
                args = st.args(outs) if callable(st.args) else dict(st.args)
            except NothingToDo as e:
                reply, stopped = str(e), True
                continue
            if st.tool in DESTRUCTIVE:
                calls, o = self._preview(st.tool, args)
                outs.append(o)
                if not o.get("ok"):
                    entry["status"] = "failed"
                    reply, stopped = f"Couldn’t {st.description[:1].lower() + st.description[1:]}: {o.get('error')}", True
                    continue
                live = [c for c in calls if c.token]
                entry["status"] = "pending" if live else "skipped"
                pending_calls += live
                pending_outs += [(c, o) for c in live]
                continue
            pre = self._before(st.tool, args)
            o = self._chunked(st.tool, args)
            outs.append(o)
            ran.append((st.tool, o))
            if not o.get("ok"):
                entry["status"] = "failed"
                reply, stopped = f"Couldn’t {st.description[:1].lower() + st.description[1:]}: {o.get('error')}", True
                continue
            entry["status"] = "done"
            undo_calls += self._undo_for(_Call(st.tool, args, extra=pre), o)

        if reply is None:
            try:
                reply = plan.reply(outs) if callable(plan.reply) else (plan.reply or "")
            except NothingToDo as e:
                reply = str(e)
            except Exception as e:  # a planner bug must not hide what already happened
                reply = f"Done, but I couldn’t describe it ({type(e).__name__})."
        if pending_outs and not pending_calls:
            reply = reply or "Nothing to change."
        resp = {"session_id": s.id, "reply": reply, "steps": steps, "results": _results(ran)}
        if pending_calls:
            p = self._make_pending(pending_calls, steps, ran)
            s.pending[p.id] = p
            resp["pending"] = {"id": p.id, "summary": p.summary, "count": p.count, "photo_ids": p.photo_ids,
                               **({"groups": p.groups} if p.groups else {})}
        u = self._remember_undo(s, undo_calls)
        if u:
            resp["undo"] = u
        return _clean(resp)

    def _chunked(self, tool: str, args: dict) -> dict:
        cap = CHUNK.get(tool)
        ids = args.get("photo_ids")
        if not cap or not isinstance(ids, list) or len(ids) <= cap:
            return self.call(tool, args)
        outs = []
        for j in range(0, len(ids), cap):
            o = self.call(tool, {**args, "photo_ids": ids[j:j + cap]})
            outs.append(o)
            if not o.get("ok"):
                break
        return _merge(outs)

    def _preview(self, tool: str, args: dict) -> tuple[list[_Call], dict]:
        args = {k: v for k, v in args.items() if k not in ("dry_run", "confirm_token")}
        cap, ids = CHUNK.get(tool), args.get("photo_ids")
        parts = [args] if not cap or not isinstance(ids, list) or len(ids) <= cap else \
            [{**args, "photo_ids": ids[j:j + cap]} for j in range(0, len(ids), cap)]
        calls, outs = [], []
        for a in parts:
            o = self.call(tool, {**a, "dry_run": True})
            outs.append(o)
            if not o.get("ok"):
                return [], o
            c = _Call(tool, a, o.get("confirm_token"), {"ids": list(o.get("photo_ids") or [])})  # what the dry run says
            if tool == "delete_album":
                c.extra["name"] = o["album"]["name"]
                c.extra["members"] = self._album_members(a["album_id"])
            calls.append(c)
        return calls, _merge(outs)

    def _make_pending(self, calls: list[_Call], steps: list[dict], ran: list[tuple[str, dict]]) -> _Pending:
        by_tool: dict[str, list[list[int]]] = {}
        for c in calls:
            by_tool.setdefault(c.tool, []).append(c.extra["ids"])
        ids = list(dict.fromkeys(i for c in calls for i in c.extra["ids"]))
        lines, count = [], 0
        for c in calls:
            if c.tool == "delete_album":
                lines.append(f"delete the album “{c.extra['name']}” (its {_plural(len(c.extra['members']), 'photo')} stay in your library)")
            elif c.tool == "purge_trash":
                lines.append("permanently empty Recently Deleted (can’t be undone)")
        if "trash_photos" in by_tool:
            n = sum(map(len, by_tool["trash_photos"]))
            count += n
            lines.insert(0, f"move {_plural(n, 'photo')} to Recently Deleted")
        if "remove_from_album" in by_tool:
            n = sum(map(len, by_tool["remove_from_album"]))
            count += n
            lines.append(f"take {_plural(n, 'photo')} out of the album (they stay in your library)")
        summary = "; ".join(lines)
        summary = summary[:1].upper() + summary[1:]
        groups = None
        cands = [o for t, o in ran if t == "find_cleanup_candidates"]
        if cands:
            want = set(ids)
            groups = []
            for x in cands[-1].get("candidates", []):
                if "keep" in x and want & set(x["remove"]):
                    groups.append({"kind": x["kind"], "keep": x["keep"], "remove": [p for p in x["remove"] if p in want],
                                   "confidence": x["confidence"], "reason": (x.get("reasons") or [""])[0]})
        return _Pending(_new_id("p"), self.clock(), calls, steps, summary + ".", count or len(ids), ids, groups or None)

    def _album_members(self, album_id: int) -> list[int]:
        ids, off = [], 0
        while off < 5000:
            o = self.call("get_album", {"album_id": album_id, "limit": 200, "offset": off})
            if not o.get("ok"):
                break
            ids += [p["id"] for p in o["photos"]]
            if len(o["photos"]) < 200:
                break
            off += 200
        return ids

    def _before(self, tool: str, args: dict) -> dict:
        """State an undo needs, captured before a non-destructive mutation runs."""
        if tool == "add_to_album" and "album_id" in args:
            return {"members": set(self._album_members(args["album_id"]))}
        if tool == "restore_photos" and isinstance(args.get("photo_ids"), list):
            reasons = {}
            ids = args["photo_ids"]
            for j in range(0, len(ids), tools.MAX_GET):
                o = self.call("get_photos", {"photo_ids": ids[j:j + tools.MAX_GET]})
                for p in o.get("photos", []):
                    reasons[p["id"]] = p.get("trash_reason") or "restored by mistake"
            return {"reasons": reasons}
        if tool == "rename_album" and "album_id" in args:
            o = self.call("get_album", {"album_id": args["album_id"], "limit": 1})
            return {"old_name": o["album"]["name"]} if o.get("ok") else {}
        return {}

    def _undo_for(self, c: _Call, o: dict) -> list[_Call]:
        t, a = c.tool, c.args
        if t == "trash_photos" and o.get("photo_ids"):
            return [_Call("restore_photos", {"photo_ids": o["photo_ids"]})]
        if t == "remove_from_album" and o.get("photo_ids"):
            return [_Call("add_to_album", {"album_id": a["album_id"], "photo_ids": o["photo_ids"]})]
        if t == "delete_album":
            members = c.extra.get("members") or []
            return [_Call("create_album", {"name": c.extra.get("name") or o["album"]["name"],
                                           **({"photo_ids": members[:tools.MAX_ALBUM_BATCH]} if members else {})})]
        if t == "create_album":
            return [_Call("delete_album", {"album_id": o["album"]["id"]}, extra={"name": o["album"]["name"]})]
        if t == "add_to_album":
            before = c.extra.get("members", set())
            new = [i for i in a.get("photo_ids", []) if i not in before and i not in set(o.get("skipped_trashed") or [])
                   and i not in set(o.get("skipped_missing") or [])]
            return [_Call("remove_from_album", {"album_id": a["album_id"], "photo_ids": new},
                          extra={"name": o["album"]["name"]})] if new and o.get("added") else []
        if t == "restore_photos" and o.get("photo_ids"):
            by_reason: dict[str, list[int]] = {}
            for i in o["photo_ids"]:
                by_reason.setdefault(c.extra.get("reasons", {}).get(i, "restored by mistake"), []).append(i)
            return [_Call("trash_photos", {"photo_ids": ids[j:j + tools.MAX_TRASH], "reason": r})
                    for r, ids in by_reason.items() for j in range(0, len(ids), tools.MAX_TRASH)]
        if t == "rename_album" and c.extra.get("old_name"):
            return [_Call("rename_album", {"album_id": a["album_id"], "new_name": c.extra["old_name"]})]
        return []  # purge_trash and reads: nothing to undo

    def _remember_undo(self, s: _Session, calls: list[_Call]) -> dict | None:
        if not calls:
            return None
        u = _Undo(_new_id("u"), _undo_label(calls[::-1]), calls[::-1])
        s.undos[u.id] = u
        while len(s.undos) > MAX_UNDO:
            s.undos.popitem(last=False)
        return {"id": u.id, "label": u.label}


# ---------------------------------------------------------------- formatting helpers

def _merge(outs: list[dict]) -> dict:
    if len(outs) == 1:
        return outs[0]
    bad = next((o for o in outs if not o.get("ok")), None)
    if bad:
        return bad
    m = dict(outs[0])
    for o in outs[1:]:
        for k, v in o.items():
            if isinstance(v, bool) or k == "confirm_token":
                continue
            if isinstance(v, int) and isinstance(m.get(k), int):
                m[k] += v
            elif isinstance(v, list) and isinstance(m.get(k), list):
                m[k] = m[k] + v
    m.pop("confirm_token", None)
    return m


def _group(done: list[tuple[_Call, dict]]) -> list[tuple[_Call, dict]]:
    """Chunks of the same tool read as one sentence."""
    out: list[tuple[_Call, dict]] = []
    for c, o in done:
        if out and out[-1][0].tool == c.tool and c.tool in ("trash_photos", "restore_photos"):
            out[-1] = (out[-1][0], _merge([out[-1][1], o]))
        else:
            out.append((c, o))
    return out


def _done_text(tool: str, o: dict) -> str:
    n = len(o.get("photo_ids") or [])
    name = (o.get("album") or {}).get("name")
    return {
        "trash_photos": f"{_plural(o.get('trashed', n), 'photo')} moved to Recently Deleted",
        "restore_photos": f"{_plural(o.get('restored', n), 'photo')} restored from Recently Deleted",
        "remove_from_album": f"{_plural(o.get('removed', n), 'photo')} taken out of “{name}”",
        "add_to_album": f"{_plural(o.get('added', 0), 'photo')} put back in “{name}”",
        "delete_album": f"album “{name}” deleted (its photos stay in your library)",
        "create_album": f"album “{name}” recreated" + (f" with {_plural(o.get('added', 0), 'photo')}" if o.get("added") else ""),
        "purge_trash": f"{_plural(o.get('purged', n), 'photo')} permanently removed from Recently Deleted",
        "rename_album": f"album renamed back to “{name}”",
    }.get(tool, tool.replace("_", " "))


def _undo_label(calls: list[_Call]) -> str:
    c = calls[0]
    n = sum(len(x.args.get("photo_ids") or []) for x in calls if x.tool == c.tool)
    return {
        "restore_photos": f"Restore {_plural(n, 'photo')}",
        "trash_photos": f"Move {_plural(n, 'photo')} back to Recently Deleted",
        "delete_album": f"Delete the album “{c.extra.get('name', '')}”",
        "remove_from_album": f"Take {_plural(n, 'photo')} back out of “{c.extra.get('name', '')}”",
        "add_to_album": f"Put {_plural(n, 'photo')} back in the album",
        "create_album": f"Recreate the album “{c.args.get('name')}”",
        "rename_album": f"Rename back to “{c.args.get('new_name')}”",
    }.get(c.tool, "Undo")


def _results(ran: list[tuple[str, dict]]) -> dict | None:
    """What the wall should show: the last read that produced photos (or albums / cleanup groups)."""
    res = None
    for tool, o in ran:
        if not o.get("ok"):
            continue
        if tool == "search_photos":
            res = {"photo_ids": _ids_of(o), "view": "photos"}
        elif tool == "get_album":
            res = {"photo_ids": _ids_of(o), "view": "album", "album": o["album"]}
        elif tool == "list_trash":
            res = {"photo_ids": _ids_of(o), "view": "trash", "total": o["total"]}
        elif tool == "list_albums":
            res = {"photo_ids": [a["cover"] for a in o["albums"] if a.get("cover")], "view": "albums", "albums": o["albums"]}
        elif tool == "find_cleanup_candidates" and o.get("candidates"):
            groups = [{"kind": x["kind"], "keep": x["keep"], "remove": x["remove"]} for x in o["candidates"] if "keep" in x]
            res = {"photo_ids": list(dict.fromkeys(p for x in o["candidates"] for p in x["photo_ids"])), "view": "cleanup",
                   **({"groups": groups} if groups else {})}
    return res


def _clean(resp: dict) -> dict:
    return {k: v for k, v in resp.items() if v is not None}


if __name__ == "__main__":
    import json
    import sys
    tools.init()
    ex = Executor()
    print(json.dumps(ex.message(" ".join(sys.argv[1:]) or "what can you do"), indent=1, ensure_ascii=False)[:4000])
