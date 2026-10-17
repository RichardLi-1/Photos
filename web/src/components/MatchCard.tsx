import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { api, type MatchContext } from "../api";
import { Glass } from "./Glass";

type Props = { id: number; anchor: DOMRect; ctx: MatchContext; onClose: () => void; onMore: (id: number, place?: boolean) => void };

const W = 264;

/** Tiny card beside a clicked photo: what it has in common with the current search. */
export function MatchCard({ id, anchor, ctx, onClose, onMore }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const { data, isLoading, error } = useQuery({
    queryKey: ["explain", id, ctx.q, ctx.ref, ctx.tags?.join(",")],
    queryFn: () => api.explain(id, ctx),
    staleTime: Infinity,
  });

  useEffect(() => {
    const down = (e: MouseEvent) => !el.current?.contains(e.target as Node) && onClose();
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("mousedown", down);
    window.addEventListener("keydown", key);
    window.addEventListener("scroll", onClose, { passive: true });
    return () => {
      window.removeEventListener("mousedown", down);
      window.removeEventListener("keydown", key);
      window.removeEventListener("scroll", onClose);
    };
  }, [onClose]);

  const left = Math.min(Math.max(12, anchor.left + anchor.width / 2 - W / 2), window.innerWidth - W - 12);
  const below = anchor.bottom + 170 < window.innerHeight - 120;
  const style = below ? { left, top: anchor.bottom - 24 } : { left, top: anchor.top + 24, transform: "translateY(-100%)" };

  const mode = data?.mode ?? "";
  const label = mode.endsWith("-overall") ? "Overall look"
    : mode === "text" ? `Matches “${ctx.q}”`
    : mode === "similar" ? "In common with that photo"
    : mode === "photo" ? "In common with your photo"
    : "In this photo";

  return (
    <div ref={el} className="fixed z-40" style={{ width: W, ...style }}>
      <Glass variant="regular" radius={20} bezel={14} depth={16} zoom={1} pillow={0} frost={14} className="px-4 py-3 text-[13px] text-black/85 dark:text-white/90">
        {isLoading && <div className="text-black/50 dark:text-white/60">Looking…</div>}
        {error && <div>Couldn’t explain this one.</div>}
        {data && (
          <>
            <div className="mb-1.5 truncate text-[11px] font-medium tracking-wide text-black/50 dark:text-white/55">{label}</div>
            {data.shared.length > 0 ? (
              <div className="flex flex-wrap gap-1">
                {data.shared.map((t) => (
                  <span key={t} className="rounded-full bg-white/55 px-2 py-0.5 dark:bg-white/15">{t}</span>
                ))}
              </div>
            ) : <div className="text-black/50 dark:text-white/60">Nothing specific — mostly overall look.</div>}
            {data.text_hits.length > 0 && <div className="mt-1.5">“{data.text_hits.join(", ")}” is written in the photo</div>}
            {data.place_hits.length > 0 && <div className="mt-1.5">“{data.place_hits.join(", ")}” is in the place name</div>}
            {(data.place || data.date) && (
              <div className="mt-2 truncate text-[11px] text-black/50 dark:text-white/55">
                {[data.place, data.date?.slice(0, 10)].filter(Boolean).join(" · ")}
              </div>
            )}
            {data.credit && (
              <div className="mt-1 truncate text-[11px] text-black/50 dark:text-white/55" title={`Photo: ${data.credit}`}>
                📷 {data.credit} · <a href="/credits" className="underline">credits</a>
              </div>
            )}
            <div className="mt-2 flex gap-4">
              <button onClick={() => onMore(id)} className="font-medium text-[var(--color-accent)]">More like this →</button>
              <button onClick={() => onMore(id, true)} title="Other photos of this same room or place (our trained model)"
                className="font-medium text-[var(--color-accent)]">Same place →</button>
            </div>
          </>
        )}
      </Glass>
    </div>
  );
}
