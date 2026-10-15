// Client for the local photosearch API (search/server.py). Same-origin only; nothing leaves the machine.

export type Photo = {
  id: number;
  album: string;
  date: string | null;
  place: string | null;
  ocr: string | null;
  score: number | null;
};

/** What /api/search's query parser understood (photosearch/query.py): hard filters + the remaining visual text. */
export type Parsed = {
  text: string;
  after?: string;
  before?: string;   // exclusive
  dates?: [string | null, string | null][];
  place?: string;    // "A|B" = any of them
  album?: string;
  has_text?: boolean;
  near?: [number, number, number];
};
export type SearchResult = { results: Photo[]; embedded: number; qtags?: string[]; parsed?: Parsed | null };
export type Explain = {
  mode: string;
  shared: string[];
  text_hits: string[];
  place_hits: string[];
  place: string | null;
  date: string | null;
};
export type MatchContext = { q?: string; ref?: number; tags?: string[] };
export type Page = { ids: number[]; total: number };
export type Stats = { photos: number; embedded: number; albums: Record<string, number>; date_range?: [string | null, string | null] };

/** /api/ask: a local VLM plans the search, then looks at the top hits and keeps the true matches (search/photosearch/ask.py). */
export type AskPlan = {
  queries: string[];
  text_in_photo?: string | string[] | null;
  after?: string | null;
  before?: string | null;
  place?: string | null;
  answer_type?: "photos" | "answer" | "count";
  image_role?: "same_place" | "similar" | "modify";
};
export type AskResult = {
  plan: AskPlan;
  answer: string | null;
  results: (Photo & { verified: boolean })[];
  timing: { "plan+recall_s": number; verify_s: number };
};
/** search/api_agent.py: the photo agent (mock planner today). Destructive steps wait for /confirm; mutations get an undo. */
export type AgentStep = { tool: string; description: string; status: "done" | "pending" | "skipped" | "failed" | "cancelled" | "partial" };
export type AgentGroup = { kind: string; keep: number; remove: number[]; confidence?: number; reason?: string };
export type AgentAlbum = { id: number; name: string; count: number; cover: number | null };
export type AgentReply = {
  session_id: string;
  reply: string;
  steps: AgentStep[];
  results?: {
    photo_ids: number[];
    view?: "photos" | "album" | "albums" | "trash" | "cleanup" | "affected";
    groups?: AgentGroup[];
    albums?: AgentAlbum[];
    album?: { id: number; name: string; count: number };
    total?: number;
  };
  pending?: { id: string; summary: string; count: number; photo_ids: number[]; groups?: AgentGroup[] };
  undo?: { id: string; label: string };
};
export type AlbumPage = { album: { id: number; name: string; count: number }; photos: { id: number }[] };

/** search/api_ask.py: is the model loaded yet, and is Ask queued behind another on-device ML job? */
export type AskStatus = { loaded: boolean; queued: boolean };

export type Query =
  | { kind: "text"; q: string }
  | { kind: "image"; file: File; text: string }
  | { kind: "similar"; id: number; place?: boolean };  // place: our trained "same room/place" model

const K = 300;

export class HttpError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new HttpError(body.detail ?? `HTTP ${res.status}`, res.status);
  return body as T;
}

export const api = {
  stats: () => fetch("/api/stats").then(json<Stats>),
  browse: (offset: number, limit: number) => fetch(`/api/browse?offset=${offset}&limit=${limit}`).then(json<Page>),
  thumb: (id: number, w = 384) => `/photo/${id}?w=${w}`,
  explain(id: number, c: MatchContext) {
    const p = new URLSearchParams();
    if (c.q) p.set("q", c.q);
    if (c.ref !== undefined) p.set("ref", String(c.ref));
    if (c.tags?.length) p.set("tags", c.tags.join(","));
    return fetch(`/api/explain/${id}?${p}`).then(json<Explain>);
  },

  search(q: Query): Promise<SearchResult> {
    if (q.kind === "text") return fetch(`/api/search?${new URLSearchParams({ q: q.q, k: String(K) })}`).then(json<SearchResult>);
    if (q.kind === "similar") return fetch(`/api/similar/${q.id}?k=${K}${q.place ? "&mode=place" : ""}`).then(json<SearchResult>);
    const fd = new FormData();
    fd.append("file", q.file);
    if (q.text) fd.append("text", q.text);
    fd.append("k", String(K));
    return fetch("/api/search_image", { method: "POST", body: fd }).then(json<SearchResult>);
  },

  ask(question: string, file: File | null, signal?: AbortSignal): Promise<AskResult> {
    if (!file) {
      return fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }), signal })
        .then(json<AskResult>);
    }
    const fd = new FormData();
    fd.append("question", question);
    fd.append("file", file);
    return fetch("/api/ask", { method: "POST", body: fd, signal }).then(json<AskResult>);
  },
  askStatus: () => fetch("/api/ask/status").then(json<AskStatus>),

  agent: {
    message: (text: string, session_id?: string) => post<AgentReply>("/api/agent/message", { text, session_id }),
    confirm: (session_id: string, pending_id: string) => post<AgentReply>("/api/agent/confirm", { session_id, pending_id }),
    cancel: (session_id: string, pending_id: string) => post<AgentReply>("/api/agent/cancel", { session_id, pending_id }),
    undo: (session_id: string, undo_id: string) => post<AgentReply>("/api/agent/undo", { session_id, undo_id }),
  },
  album: (id: number) => fetch(`/api/albums/${id}?limit=200`).then(json<AlbumPage>),
};

function post<T>(url: string, body: unknown): Promise<T> {
  return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(json<T>);
}

export function queryKey(q: Query) {
  return q.kind === "image" ? ["image", q.file.name, q.file.size, q.file.lastModified, q.text]
    : q.kind === "similar" ? [q.kind, q.id, !!q.place] : [q.kind, q.q];
}
