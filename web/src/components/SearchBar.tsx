import { useEffect, useRef, useState, type ReactNode } from "react";
import { isCommand } from "../agent";
import { isQuestion } from "../ask";
import { Glass } from "./Glass";

type Props = {
  text: string;
  onText: (text: string) => void;
  placeholder: string;
  status: ReactNode;
  image: File | null;
  onSearch: (text: string) => void;
  onImage: (f: File | null) => void;
  onClear: () => void;
  agentMode: boolean;
  onAgentMode: (on: boolean) => void;
  links?: ReactNode;     // small text links above the pill (Albums · Recently Deleted)
  children?: ReactNode;  // shown above the pill (the Ask answer / agent card)
};

/** Floating liquid-glass search pill. Enter searches (or asks, for a question); Esc clears; drop or pick a photo to search by image. */
export function SearchBar({ text, onText, placeholder, status, image, onSearch, onImage, onClear, agentMode, onAgentMode, links, children }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const file = useRef<HTMLInputElement>(null);
  const preview = useObjectUrl(image);

  useEffect(() => {
    // Type anywhere to search: the first printable keystroke moves focus to the pill and lands in it.
    const key = (e: KeyboardEvent) => {
      const el = input.current, active = document.activeElement;
      if (!el || active === el || e.metaKey || e.ctrlKey || e.altKey || e.key.length !== 1) return;
      if (active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement || (active as HTMLElement)?.isContentEditable) return;
      el.focus();
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);

  const clear = () => { onText(""); onImage(null); onClear(); };
  const toAgent = agentMode || isCommand(text);

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-[max(24px,env(safe-area-inset-bottom))] z-30 flex flex-col items-center gap-2 px-4">
      {(status || links) && (
        <div className="relative flex w-full max-w-[720px] justify-center">
          {status && (
            <div className="pointer-events-auto flex max-w-full flex-wrap items-center justify-center gap-x-1.5 gap-y-1 rounded-full bg-black/45 px-3 py-1 text-xs text-white backdrop-blur-md">{status}</div>
          )}
          {links && !status && <div className="pointer-events-auto absolute right-2 bottom-0 flex gap-1">{links}</div>}
        </div>
      )}
      {children}
      <Glass className="pointer-events-auto flex w-full max-w-[720px] items-center gap-3 px-6 py-3 sm:px-8 sm:py-4">
        {preview && (
          <img src={preview} alt="" className="h-10 w-10 shrink-0 rounded-xl object-cover ring-1 ring-white/70" />
        )}
        <form className="min-w-0 flex-1" onSubmit={(e) => { e.preventDefault(); if (text.trim() || image) onSearch(text.trim()); }}>
          <input ref={input} value={text} onChange={(e) => onText(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && clear()}
            placeholder={agentMode ? "Tell the agent: “clean up duplicates”" : image ? "Add words to refine (optional)" : placeholder}
            aria-label="Search photos"
            style={{ textShadow: "0 0 12px rgb(255 255 255 / 0.7)" }}
            className="w-full bg-transparent text-xl font-normal tracking-tight text-black/85 outline-none placeholder:text-black/45 dark:text-white/90 dark:placeholder:text-white/55 sm:text-[34px]" />
        </form>
        {!toAgent && isQuestion(text) && (
          <span title="Enter asks the on-device model" className="shrink-0 rounded-full bg-accent/15 px-2.5 py-1 text-sm font-medium text-accent">Ask ↵</span>
        )}
        <button onClick={() => { onAgentMode(!agentMode); input.current?.focus(); }} aria-pressed={agentMode}
          title={agentMode ? "Agent mode: Enter sends to the photo agent (click to turn off)" : toAgent ? "Enter sends this to the photo agent" : "Photo agent: albums, cleanup, trash"}
          className={`shrink-0 rounded-full px-2.5 py-1 text-sm font-medium transition-colors ${agentMode ? "bg-accent text-white"
            : toAgent ? "bg-white/75 text-accent ring-1 ring-accent/40 dark:bg-black/55" : "text-black/50 ring-1 ring-black/15 hover:bg-white/40 dark:text-white/60 dark:ring-white/20"}`}>
          {toAgent && text.trim() ? "Agent ↵" : "Agent"}
        </button>
        {(text || image) && (
          <button onClick={clear} aria-label="Clear" className="shrink-0 rounded-full p-2 text-black/75 hover:bg-white/40 dark:text-white/85">✕</button>
        )}
        <button onClick={() => file.current?.click()} aria-label="Search with a photo" title="Search with a photo"
          className="shrink-0 rounded-full p-2 text-black/75 hover:bg-white/40 dark:text-white/85">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="5" width="18" height="15" rx="3" /><circle cx="12" cy="12.5" r="3.5" /><path d="M8 5l1.5-2h5L16 5" />
          </svg>
        </button>
        <input ref={file} type="file" accept="image/*" hidden
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onImage(f); e.target.value = ""; }} />
      </Glass>
    </div>
  );
}

function useObjectUrl(f: File | null) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!f) return setUrl(null);
    const u = URL.createObjectURL(f);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [f]);
  return url;
}
