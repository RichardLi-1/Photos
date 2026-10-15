"""Skeleton for plugging a LOCAL LLM into the photo tools. The agent loop itself is yours to write.

    .venv/bin/python agent_stub.py "clean up my blurry duplicates"

Tools run in-process here (photosearch.tools); over HTTP use GET /api/tools/schema + POST /api/tools/{name}.
"""
import json
import sys

from photosearch import tools

SCHEMAS = tools.schemas()  # [{name, description, input_schema}] -> pass as the tool list to your local model
DESTRUCTIVE = tools.DESTRUCTIVE  # trash_photos, purge_trash, delete_album, remove_from_album


def confirm_with_user(name: str, preview: dict) -> bool:
    """REQUIRED before any destructive call with dry_run=false. Show the dry-run preview; True only on explicit yes."""
    print(json.dumps({k: preview.get(k) for k in ("would_trash", "would_purge", "would_remove", "photos_in_album",
                                                   "album", "photo_ids", "reason", "note") if k in preview}, indent=1))
    return input(f"Run {name} for real? [y/N] ").strip().lower() == "y"


def execute(name: str, args: dict) -> dict:
    """Run one tool call from the model, enforcing preview -> human confirm -> execute for destructive tools."""
    if name in DESTRUCTIVE and args.get("dry_run") is False:
        preview = tools.call(name, {**args, "dry_run": True, "confirm_token": None}, actor="agent")
        if not preview.get("ok") or not preview.get("confirm_token"):
            return preview  # nothing to do, or invalid
        if not confirm_with_user(name, preview):
            return {"ok": False, "error": "the user declined; nothing was changed"}
        args = {**args, "confirm_token": preview["confirm_token"]}  # token binds the call to exactly what was shown
    return tools.call(name, args, actor="agent")


def run_agent(user_message: str):
    # TODO(user): your agent loop / planning / confirmation policy here.
    # e.g. load a local model (mlx-lm / llama.cpp), send `user_message` + SCHEMAS, then repeat:
    #   tool_calls = model(messages, tools=SCHEMAS); results = [execute(c.name, c.args) for c in tool_calls]
    #   append results to messages; stop when the model answers without tool calls.
    raise NotImplementedError("agent loop not implemented yet")


if __name__ == "__main__":
    tools.init()  # standalone: opens data/index read-only for search, plus its own write connection for tools
    run_agent(" ".join(sys.argv[1:]) or "find photos I probably don't need")
