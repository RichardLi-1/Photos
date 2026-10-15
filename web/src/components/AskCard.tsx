import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, HttpError, type AskPlan, type AskResult } from "../api";
import { Glass } from "./Glass";

type Props = {
  data?: AskResult & { ms: number };
  error: Error | null;
  pending: boolean;
  hasImage: boolean;
  onRetry: () => void;
  onClose: () => void;
};

const CHECKED = 24;  // Asker.ask(verify_n): how many top hits the model looks at

/** Glass card above the pill: the local model's answer, how it searched, and how long it took. Esc or ✕ closes. */
export function AskCard({ data, error, pending, hasImage, onRetry, onClose }: Props) {
  const secs = useElapsed(pending);
  const status = useQuery({ queryKey: ["askStatus"], queryFn: api.askStatus, enabled: pending, refetchInterval: 1500, retry: false, gcTime: 0 });

  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);

  const busy = error instanceof HttpError && error.status === 503;
  const verified = data?.results.filter((r) => r.verified).length ?? 0;
  const progress = status.data?.queued ? "Waiting for another on-device job to finish…"
    : status.data && !status.data.loaded ? "Loading the on-device model (about 3 GB, first question only)…"
    : "Looking through your photos…";

  return (
    <Glass variant="regular" radius={24} bezel={16} depth={16} zoom={1} pillow={0} frost={14}
      className="pointer-events-auto w-full max-w-[720px] px-5 py-4 text-[14px] text-black/85 sm:px-6 dark:text-white/90">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1" aria-live="polite">
          {pending && (
            <>
              <div className="flex items-center gap-2.5 text-[16px]">
                <span className="ask-breathe size-2 shrink-0 rounded-full bg-accent" />
                <span className="min-w-0 flex-1">{progress}</span>
                <span className="shrink-0 tabular-nums text-black/45 dark:text-white/50">{secs} s</span>
              </div>
              <div className="mt-1 pl-[18px] text-[12px] text-black/50 dark:text-white/55">
                Answers take 10–60 seconds. Everything runs on this Mac.
              </div>
            </>
          )}
          {!pending && busy && (
            <>
              <div className="text-[16px]">This Mac is busy right now.</div>
              <div className="mt-1 text-[12px] text-black/50 dark:text-white/55">{error.message}</div>
              <button onClick={onRetry} className="mt-2 font-medium text-accent">Try again</button>
            </>
          )}
          {!pending && error && !busy && (
            <>
              <div className="text-[16px]">Couldn’t answer that.</div>
              <div className="mt-1 text-[12px] text-black/50 dark:text-white/55">{error.message}</div>
              <button onClick={onRetry} className="mt-2 font-medium text-accent">Try again</button>
            </>
          )}
          {!pending && data && (
            <>
              <div className="max-h-[28vh] overflow-auto text-[17px] leading-snug">
                {data.answer ?? (verified ? `Found ${verified} matching photo${verified === 1 ? "" : "s"}.`
                  : "I couldn’t confirm a match. These are the closest photos.")}
              </div>
              <div className="mt-2 truncate text-[12px] text-black/55 dark:text-white/60" title={how(data.plan, hasImage)}>
                <span className="font-medium">How I searched:</span> {how(data.plan, hasImage)}
              </div>
              <div className="mt-0.5 text-[12px] text-black/45 dark:text-white/50"
                title={`plan + search ${data.timing["plan+recall_s"]} s · checking ${data.timing.verify_s} s`}>
                {verified} of {Math.min(CHECKED, data.results.length)} checked photos match ✓ · {Math.round(data.ms / 1000)} s · on-device
              </div>
            </>
          )}
        </div>
        <button onClick={onClose} aria-label="Close answer" className="-mr-2 -mt-1 shrink-0 rounded-full p-1.5 text-black/60 hover:bg-white/40 dark:text-white/70">✕</button>
      </div>
    </Glass>
  );
}

/** The plan in one line: the visual queries, then any literal text / dates / place / photo role. */
function how(p: AskPlan, hasImage: boolean) {
  const bits = [p.queries.map((q) => `“${q}”`).join(", ")];
  const text = [p.text_in_photo ?? []].flat().filter(Boolean).join(" ");
  if (text) bits.push(`text “${text}”`);
  if (p.after && p.before) bits.push(`${p.after} → ${p.before}`);
  else if (p.after) bits.push(`after ${p.after}`);
  else if (p.before) bits.push(`before ${p.before}`);
  if (p.place) bits.push(`in ${p.place}`);
  if (hasImage) bits.push(p.image_role === "same_place" ? "same place as your photo" : p.image_role === "modify" ? "your photo, changed" : "like your photo");
  return bits.join(" · ");
}

function useElapsed(running: boolean) {
  const [s, setS] = useState(0);
  useEffect(() => {
    if (!running) return;
    const t0 = Date.now();
    setS(0);
    const iv = setInterval(() => setS(Math.round((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(iv);
  }, [running]);
  return s;
}
