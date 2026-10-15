import { useEffect, useState } from "react";
import { Virtuoso } from "react-virtuoso";
import { Story } from "./Story";
import { GAP, Row, type Pick } from "./Tile";

/** Same columns the old CSS grid gave: auto-fill minmax(150px, or 220px from 640px up, 1fr) with 4px gaps. */
function measure() {
  const w = document.documentElement.clientWidth, h = window.innerHeight;
  const min = w >= 640 ? 220 : 150;
  const cols = Math.max(1, Math.floor((w + GAP) / (min + GAP)));
  return { w, h, cols, tile: (w - (cols - 1) * GAP) / cols };
}

function useLayout() {
  const [l, setL] = useState(measure);
  useEffect(() => {
    const on = () => setL((o) => { const n = measure(); return n.w === o.w && n.h === o.h ? o : n; });
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return l;
}

type Props = { ids: number[]; onEnd?: () => void; onPick: Pick; verified?: Set<number>; story?: { photos?: number } | null };

/**
 * Edge-to-edge, window-scrolling, virtualized photo wall, as rows of N tiles (N from the width).
 * In browse mode, after ~2.5 screens one full-row item is the Story: a screenful of tiles whose middle one opens into the product story.
 */
export function Wall({ ids, onEnd, onPick, verified, story }: Props) {
  const { w, h, cols, tile } = useLayout();
  const rows = Math.ceil(ids.length / cols);
  const at = Math.max(2, Math.round((2.5 * h) / (tile + GAP)));  // story's position, in rows
  const span = Math.ceil(h / (tile + GAP));                       // rows of photos the story stage shows
  const hasStory = !!story && rows > at + span;
  const count = hasStory ? rows - span + 1 : rows;
  const rowOf = (i: number) => (hasStory && i > at ? i + span - 1 : i);
  const isStory = (i: number) => hasStory && i === at;

  return (
    <Virtuoso
      useWindowScroll
      totalCount={count}
      increaseViewportBy={800}
      endReached={onEnd}
      computeItemKey={(i) => (isStory(i) ? "story" : rowOf(i))}
      itemContent={(i) => isStory(i)
        ? <Story ids={ids.slice(at * cols, (at + span) * cols)} cols={cols} tile={tile} width={w} height={h} photos={story?.photos} onPick={onPick} />
        : <Row ids={ids.slice(rowOf(i) * cols, (rowOf(i) + 1) * cols)} cols={cols} onPick={onPick} verified={verified} />}
    />
  );
}
