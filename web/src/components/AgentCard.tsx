import { useEffect, useRef, useState } from "react";
import { api, type AgentAlbum, type AgentGroup, type AgentReply, type AgentStep } from "../api";
import { Glass } from "./Glass";

type Props = {
  data: AgentReply | null;
  busy: boolean;
  error: string | null;
  albumId?: number | null;
  onConfirm: () => void;
  onCancel: () => void;
  onUndo: () => void;
  onSend: (text: string) => void;
  onAlbum: (a: AgentAlbum) => void;
  onClose: () => void;
};

const ICON: Record<AgentStep["status"], string> = { done: "✓", pending: "⏳", failed: "✕", skipped: "–", cancelled: "–", partial: "◐" };
const MAX_THUMBS = 36;

/** Glass card above the pill: the agent's reply, its step checklist, and Confirm / Cancel / Undo. Esc or ✕ closes. */
export function AgentCard({ data, busy, error, albumId, onConfirm, onCancel, onUndo, onSend, onAlbum, onClose }: Props) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);

  const pending = data?.pending;
  const albums = data?.results?.view === "albums" ? data.results.albums : undefined;
  const trashCount = data?.results?.view === "trash" ? data.results.total ?? 0 : 0;

  return (
    <Glass variant="regular" radius={24} bezel={16} depth={16} zoom={1} pillow={0} frost={14}
      className="pointer-events-auto w-full max-w-[720px] px-5 py-4 text-[14px] text-black/85 sm:px-6 dark:text-white/90">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1" aria-live="polite">
          {busy && (
            <div className="flex items-center gap-2.5 text-[16px]">
              <span className="ask-breathe size-2 shrink-0 rounded-full bg-accent" />
              <span>Working on it…</span>
            </div>
          )}
          {!busy && error && (
            <>
              <div className="text-[16px]">The agent couldn’t do that.</div>
              <div className="mt-1 text-[12px] text-black/50 dark:text-white/55">{error}</div>
            </>
          )}
          {!busy && data && (
            <>
              <div className="max-h-[22vh] overflow-auto text-[17px] leading-snug">{data.reply}</div>
              {data.steps.length > 0 && (
                <ul className="mt-2 space-y-0.5 text-[12px] text-black/55 dark:text-white/60">
                  {data.steps.map((s, i) => (
                    <li key={i} className="flex gap-1.5">
                      <span className={`w-4 shrink-0 text-center ${s.status === "done" ? "text-accent" : s.status === "failed" ? "text-red-600 dark:text-red-400" : ""}`}>{ICON[s.status]}</span>
                      <span className="min-w-0 truncate">{s.description}{s.status === "pending" ? " — waiting for you" : ""}</span>
                    </li>
                  ))}
                </ul>
              )}
              {pending && (
                <div className="mt-3">
                  <div className="text-[13px] font-medium">{pending.summary}</div>
                  <Preview ids={pending.photo_ids} groups={pending.groups} />
                  <ConfirmButtons onConfirm={onConfirm} onCancel={onCancel} count={pending.count} />
                </div>
              )}
              {albums && albums.length > 0 && (
                <div className="mt-2.5 flex flex-wrap gap-1.5">
                  {albums.map((a) => (
                    <button key={a.id} onClick={() => onAlbum(a)}
                      className={`rounded-full px-2.5 py-1 text-[13px] ${albumId === a.id ? "bg-accent text-white" : "bg-black/[0.06] hover:bg-black/10 dark:bg-white/15 dark:hover:bg-white/25"}`}>
                      {a.name} <span className="opacity-60">{a.count}</span>
                    </button>
                  ))}
                </div>
              )}
              {trashCount > 0 && !pending && (
                <button onClick={() => onSend("restore")} className="mt-2 font-medium text-accent">Restore all</button>
              )}
              {data.undo && !pending && (
                <div className="mt-3 flex items-center gap-2">
                  <button onClick={onUndo}
                    className="rounded-full bg-white/60 px-3.5 py-1.5 text-[14px] font-medium text-accent ring-1 ring-black/10 hover:bg-white/85 dark:bg-white/15 dark:ring-white/15">
                    Undo
                  </button>
                  <span className="truncate text-[12px] text-black/45 dark:text-white/50">{data.undo.label}</span>
                </div>
              )}
            </>
          )}
        </div>
        <button onClick={onClose} aria-label="Close" className="-mr-2 -mt-1 shrink-0 rounded-full p-1.5 text-black/60 hover:bg-white/40 dark:text-white/70">✕</button>
      </div>
    </Glass>
  );
}

/** Confirm gets focus so Enter confirms; a short arming delay stops a held-down Enter from confirming by accident. */
function ConfirmButtons({ onConfirm, onCancel, count }: { onConfirm: () => void; onCancel: () => void; count: number }) {
  const ok = useRef<HTMLButtonElement>(null);
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    ok.current?.focus({ preventScroll: true });
    const t = setTimeout(() => setArmed(true), 400);
    return () => clearTimeout(t);
  }, []);
  return (
    <div className="mt-3 flex items-center gap-2">
      <button ref={ok} onClick={() => armed && onConfirm()}
        className="rounded-full bg-accent px-4 py-1.5 text-[14px] font-medium text-white outline-offset-2 hover:brightness-110 focus-visible:outline-2 focus-visible:outline-accent">
        Confirm
      </button>
      <button onClick={onCancel}
        className="rounded-full px-3.5 py-1.5 text-[14px] font-medium text-black/70 ring-1 ring-black/15 hover:bg-white/50 dark:text-white/80 dark:ring-white/20">
        Cancel
      </button>
      <span className="ml-1 text-[12px] text-black/45 dark:text-white/50">{count} {count === 1 ? "photo" : "photos"} · restorable · files untouched</span>
    </div>
  );
}

/** Affected thumbnails. Duplicate groups read as keep ✓ next to remove ✕. */
function Preview({ ids, groups }: { ids: number[]; groups?: AgentGroup[] }) {
  if (!ids.length) return null;
  const vis: AgentGroup[] = [];
  let n = 0;
  for (const g of groups ?? []) {
    if (n >= MAX_THUMBS) break;
    vis.push(g);
    n += 1 + g.remove.length;
  }
  const hidden = ids.length - (vis.length ? vis.reduce((k, g) => k + g.remove.length, 0) : Math.min(ids.length, MAX_THUMBS));
  return (
    <div className="mt-2 flex items-center gap-1.5 overflow-x-auto pb-1">
      {vis.length ? vis.map((g) => (
        <div key={g.keep} title={g.reason} className="flex shrink-0 gap-0.5 rounded-[12px] bg-black/5 p-0.5 dark:bg-white/10">
          <Thumb id={g.keep} keep />
          {g.remove.map((id) => <Thumb key={id} id={id} />)}
        </div>
      )) : ids.slice(0, MAX_THUMBS).map((id) => <Thumb key={id} id={id} />)}
      {hidden > 0 && <span className="shrink-0 px-2 text-[12px] text-black/50 dark:text-white/55">+{hidden} more</span>}
    </div>
  );
}

function Thumb({ id, keep }: { id: number; keep?: boolean }) {
  return (
    <div className="relative size-14 shrink-0 overflow-hidden rounded-[10px] bg-black/5 dark:bg-white/5">
      <img src={api.thumb(id, 128)} alt="" loading="lazy" className={`h-full w-full object-cover ${keep ? "" : "opacity-75"}`} />
      <span aria-label={keep ? "keep" : "remove"}
        className={`absolute bottom-0.5 right-0.5 grid size-[18px] place-items-center rounded-full text-[11px] font-semibold text-white ring-1 ring-white/70 ${keep ? "bg-accent" : "bg-red-500"}`}>
        {keep ? "✓" : "✕"}
      </span>
    </div>
  );
}
