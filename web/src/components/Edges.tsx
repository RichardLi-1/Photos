import type { CSSProperties } from "react";

type Side = "left" | "right" | "top" | "bottom";

const DIR: Record<Side, string> = { left: "to right", right: "to left", top: "to bottom", bottom: "to top" };
const LAYERS = [
  { b: "24px", from: "0%", to: "25%" },
  { b: "12px", from: "0%", to: "45%" },
  { b: "6px", from: "10%", to: "70%" },
  { b: "2px", from: "30%", to: "100%" },
];

function Edge({ side, size }: { side: Side; size: string }) {
  const pos: CSSProperties = side === "left" || side === "right"
    ? { [side]: 0, top: 0, bottom: 0, width: size }
    : { [side]: 0, left: 0, right: 0, height: size };
  return (
    <div className="edge" style={{ ...pos, ["--dir" as string]: DIR[side] }} aria-hidden>
      {LAYERS.map((l) => (
        <i key={l.b} style={{ ["--b" as string]: l.b, ["--from" as string]: l.from, ["--to" as string]: l.to }} />
      ))}
    </div>
  );
}

/** Frosted, progressively blurred glass along the sides (and a thin top) of the viewport. */
export function Edges() {
  return (
    <>
      <Edge side="left" size="clamp(20px, 5vw, 84px)" />
      <Edge side="right" size="clamp(20px, 5vw, 84px)" />
      <Edge side="top" size="clamp(16px, 4vh, 48px)" />
      {/* no bottom edge: the search pill must refract the sharp photos, not an already-blurred strip */}
    </>
  );
}
