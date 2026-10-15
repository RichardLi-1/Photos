import { useState } from "react";
import { api } from "../api";

export const GAP = 4;

export type Pick = (id: number, rect: DOMRect) => void;

export function Tile({ id, onPick, verified, last }: { id: number; onPick: Pick; verified?: boolean; last?: boolean }) {
  const [loaded, setLoaded] = useState(false);
  return (
    <button onClick={(e) => onPick(id, e.currentTarget.getBoundingClientRect())}
      className="group relative block aspect-square w-full overflow-hidden bg-black/5 dark:bg-white/5">
      <img src={api.thumb(id)} loading="lazy" decoding="async" alt="" onLoad={() => setLoaded(true)}
        className={`h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.03] ${loaded ? "tile-img" : "opacity-0"}`} />
      {verified && (
        // bottom corner, away from the blurred top/side edges (mirrored on the last column)
        <span title="Checked by the on-device model" aria-label="verified"
          className={`absolute bottom-2 ${last ? "left-2" : "right-2"} grid size-6 place-items-center rounded-full bg-accent text-[13px] font-semibold text-white shadow-[0_1px_4px_rgb(0_0_0/0.35)] ring-1 ring-white/70`}>
          ✓
        </span>
      )}
    </button>
  );
}

export function Row({ ids, cols, onPick, verified }: { ids: number[]; cols: number; onPick: Pick; verified?: Set<number> }) {
  return (
    <div className="grid gap-1 pb-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {ids.map((id, i) => <Tile key={id} id={id} onPick={onPick} verified={verified?.has(id)} last={i === cols - 1} />)}
    </div>
  );
}
