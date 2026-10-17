import { useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { isCommand } from "./agent";
import { api, queryKey, type AgentAlbum, type AgentReply, type MatchContext, type Query } from "./api";
import { isQuestion } from "./ask";
import { AgentCard } from "./components/AgentCard";
import { AskCard } from "./components/AskCard";
import { Edges } from "./components/Edges";
import { MatchCard } from "./components/MatchCard";
import { SearchBar } from "./components/SearchBar";
import { Wall } from "./components/Wall";
import { emptyMessage, parsedChips } from "./parsed";
import { startVisits } from "./visits";

const PAGE = 400;
const STRIPES = new URLSearchParams(location.search).has("stripes");  // debug: high-contrast backdrop for the glass

type Ask = { question: string; file: File | null };
type Agent = { data: AgentReply | null; busy: boolean; error: string | null; album?: { id: number; ids: number[] } | null };

export default function App() {
  const [query, setQuery] = useState<Query | null>(() => {
    const q = new URLSearchParams(location.search).get("q");
    return q ? { kind: "text", q } : null;
  });
  const [ask, setAsk] = useState<Ask | null>(null);
  const [text, setText] = useState(query?.kind === "text" ? query.q : "");
  const [image, setImage] = useState<File | null>(null);
  const [picked, setPicked] = useState<{ id: number; rect: DOMRect } | null>(null);
  const [agent, setAgent] = useState<Agent | null>(null);
  const [agentMode, setAgentMode] = useState(false);
  const session = useRef<string | undefined>(undefined);
  const qc = useQueryClient();
  const closeCard = useCallback(() => setPicked(null), []);
  const reset = useCallback(() => { setText(""); setImage(null); setQuery(null); setAsk(null); setAgent(null); }, []);

  // The agent: one request at a time; every reply replaces the card. Mutations make cached searches stale (trash hides photos).
  const agentCall = useCallback(async (run: (sid: string | undefined) => Promise<AgentReply>, mutates = false) => {
    setAgent((a) => ({ data: a?.data ?? null, busy: true, error: null, album: a?.album }));
    try {
      const data = await run(session.current);
      session.current = data.session_id;
      setAgent({ data, busy: false, error: null });
      if (mutates || data.undo) qc.invalidateQueries({ queryKey: ["search"] });
    } catch (e) {
      setAgent({ data: null, busy: false, error: (e as Error).message });
    }
  }, [qc]);
  const sendAgent = useCallback((t: string) => {
    setQuery(null); setAsk(null);
    agentCall((sid) => api.agent.message(t, sid));
  }, [agentCall]);
  const pendingId = agent?.data?.pending?.id, undoId = agent?.data?.undo?.id;
  const agentConfirm = useCallback(() => pendingId && agentCall((sid) => api.agent.confirm(sid!, pendingId), true), [agentCall, pendingId]);
  const agentCancel = useCallback(() => pendingId && agentCall((sid) => api.agent.cancel(sid!, pendingId)), [agentCall, pendingId]);
  const agentUndo = useCallback(() => undoId && agentCall((sid) => api.agent.undo(sid!, undoId), true), [agentCall, undoId]);
  const openAlbum = useCallback(async (a: AgentAlbum) => {
    try {
      const page = await api.album(a.id);
      setAgent((s) => s && { ...s, album: { id: a.id, ids: page.photos.map((p) => p.id) } });
      window.scrollTo({ top: 0 });
    } catch (e) {
      setAgent((s) => s && { ...s, error: (e as Error).message });
    }
  }, []);

  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats, staleTime: 60_000 });
  // Visitor tracking runs only on the hosted demo (its /api/stats has `demo`); the local app never sends anything.
  useEffect(() => { if (stats.data?.demo) startVisits(); }, [stats.data]);
  const browse = useInfiniteQuery({
    queryKey: ["browse"],
    queryFn: ({ pageParam }) => api.browse(pageParam, PAGE),
    initialPageParam: 0,
    getNextPageParam: (last, pages) => (pages.length * PAGE < last.total ? pages.length * PAGE : undefined),
    staleTime: Infinity,
  });
  const search = useQuery({
    queryKey: ["search", ...(query ? queryKey(query) : [])],
    queryFn: async () => {
      const t = performance.now();
      const r = await api.search(query!);
      return { ...r, ms: performance.now() - t };
    },
    enabled: !!query,
    staleTime: Infinity,
    retry: false,
  });
  const asked = useQuery({
    queryKey: ["ask", ask?.question, ...(ask?.file ? [ask.file.name, ask.file.size, ask.file.lastModified] : [])],
    queryFn: async ({ signal }) => {
      const t = performance.now();
      const r = await api.ask(ask!.question, ask!.file, signal);
      return { ...r, ms: performance.now() - t };
    },
    enabled: !!ask,
    staleTime: Infinity,
    retry: false,
  });

  // Drop a photo anywhere to search by it.
  useEffect(() => {
    const over = (e: DragEvent) => e.preventDefault();
    const drop = (e: DragEvent) => {
      e.preventDefault();
      const f = [...(e.dataTransfer?.files ?? [])].find((f) => f.type.startsWith("image/"));
      if (f) { setImage(f); setAsk(null); setQuery({ kind: "image", file: f, text: "" }); }
    };
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => { window.removeEventListener("dragover", over); window.removeEventListener("drop", drop); };
  }, []);

  useEffect(() => {
    setPicked(null);
    window.scrollTo({ top: 0 });
    history.replaceState(null, "", query?.kind === "text" ? `?q=${encodeURIComponent(query.q)}` : location.pathname + keepFlags());
  }, [query, ask, agent?.data]);

  const answer = ask ? asked.data : undefined;
  const verified = useMemo(() => new Set(answer?.results.filter((r) => r.verified).map((r) => r.id)), [answer]);

  const ctx: MatchContext = answer ? { q: answer.plan.queries[0] }
    : query?.kind === "text" ? { q: query.q }
    : query?.kind === "similar" ? { ref: query.id }
    : query?.kind === "image" ? { tags: search.data?.qtags }
    : {};

  // While the agent card is open the wall shows what it's about: an opened album, the photos a pending action
  // affects, or the reply's results. While a request runs (or for replies without photos) the wall stays put.
  const agentIds = agent?.album?.ids ?? agent?.data?.pending?.photo_ids ?? agent?.data?.results?.photo_ids;
  const [lastAgentIds, setLastAgentIds] = useState<number[] | undefined>(undefined);
  useEffect(() => { if (!agent) setLastAgentIds(undefined); else if (agentIds?.length) setLastAgentIds(agentIds); }, [agent, agentIds]);
  const shownAgentIds = agent ? (agentIds?.length ? agentIds : lastAgentIds) : undefined;

  // While an answer is pending the current wall stays put; then it shows the answer's results, verified first.
  const ids = shownAgentIds ?? (answer ? answer.results.map((r) => r.id)
    : query ? (search.data?.results.map((r) => r.id) ?? [])
    : (browse.data?.pages.flatMap((p) => p.ids) ?? []));
  const browsing = !query && !answer && !shownAgentIds;
  const n = stats.data?.photos;
  const chips = parsedChips(search.data?.parsed);
  const status = !query || agent ? null
    : search.isFetching ? "Searching…"
    : search.error ? `Search failed: ${(search.error as Error).message}`
    : search.data && !search.data.results.length ? emptyMessage(search.data.parsed, stats.data?.date_range)
    : search.data ? <>{chips.map((c) => <Chip key={c}>{c}</Chip>)}<span>{search.data.results.length} best matches · {Math.round(search.data.ms)} ms · on-device</span></>
    : null;

  return (
    <main className="min-h-dvh">
      {STRIPES && <div className="fixed inset-0 bg-[repeating-linear-gradient(90deg,#111_0_24px,#fff_24px_48px),repeating-linear-gradient(0deg,transparent_0_46px,#e33_46px_48px)]" />}
      {!STRIPES && <Wall ids={ids} onEnd={browsing ? () => browse.hasNextPage && browse.fetchNextPage() : undefined}
        verified={answer && !shownAgentIds ? verified : undefined} story={browsing && !ask && !agent ? { photos: n } : null}
        onPick={(id, rect) => setPicked({ id, rect })} />}
      <Edges />
      {picked && (
        <MatchCard key={picked.id} id={picked.id} anchor={picked.rect} ctx={ctx} onClose={closeCard}
          onMore={(id, place) => { setPicked(null); setImage(null); setAsk(null); setQuery({ kind: "similar", id, place }); }} />
      )}
      <SearchBar
        text={text}
        onText={setText}
        placeholder={n ? `Try it on ${n.toLocaleString()} public images` : "Search photos"}
        status={status}
        image={image}
        onSearch={(t) => {
          if (t && (agentMode || isCommand(t))) sendAgent(t);
          else if (isQuestion(t)) { setAgent(null); setQuery(null); setAsk({ question: t, file: image }); }
          else { setAgent(null); setAsk(null); setQuery(image ? { kind: "image", file: image, text: t } : { kind: "text", q: t }); }
        }}
        onImage={(f) => { setImage(f); if (f) { setAgent(null); setAsk(null); setQuery({ kind: "image", file: f, text: "" }); } }}
        onClear={reset}
        agentMode={agentMode}
        onAgentMode={setAgentMode}
        links={!ask && !agent && (
          <>
            <LinkChip onClick={() => sendAgent("show my albums")}>Albums</LinkChip>
            <LinkChip onClick={() => sendAgent("show trash")}>Recently Deleted</LinkChip>
          </>
        )}
      >
        {agent && (
          <AgentCard data={agent.data} busy={agent.busy} error={agent.error} albumId={agent.album?.id}
            onConfirm={agentConfirm} onCancel={agentCancel} onUndo={agentUndo} onSend={sendAgent} onAlbum={openAlbum}
            onClose={reset} />
        )}
        {ask && (
          <AskCard key={ask.question} data={asked.data} error={asked.error} pending={asked.isFetching}
            hasImage={!!ask.file} onRetry={() => asked.refetch()} onClose={reset} />
        )}
      </SearchBar>
    </main>
  );
}

function Chip({ children }: { children: string }) {
  return <span className="rounded-full bg-white/20 px-2 py-px">{children}</span>;
}

function LinkChip({ onClick, children }: { onClick: () => void; children: string }) {
  return (
    <button onClick={onClick}
      className="rounded-full bg-black/45 px-3 py-1 text-xs text-white/90 backdrop-blur-md hover:bg-black/60 hover:text-white">
      {children}
    </button>
  );
}

/** Debug flags (?stripes, ?noglass) survive the URL rewrite. */
function keepFlags() {
  const p = new URLSearchParams(location.search), keep = new URLSearchParams();
  for (const k of ["stripes", "noglass"]) if (p.has(k)) keep.set(k, "");
  const s = keep.toString().replace(/=(&|$)/g, "$1");
  return s ? `?${s}` : "";
}
