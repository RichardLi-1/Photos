import { useEffect, useLayoutEffect, useRef } from "react";
import { api } from "../api";
import { Glass } from "./Glass";
import { GAP, Tile, type Pick } from "./Tile";

type Props = { ids: number[]; cols: number; tile: number; width: number; height: number; photos?: number; onPick: Pick };

const REDUCED = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;

const clamp = (x: number) => Math.min(1, Math.max(0, x));
const ease = (x: number) => x * x * (3 - 2 * x);  // smoothstep
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/**
 * One full-row item of the wall: a screenful of ordinary-looking tiles. As it scrolls into view the middle tile opens
 * (scroll-linked) into a full-bleed photo under a glass panel with the product story, holds while the stage is pinned,
 * then folds back into a tile before the wall carries on. Transforms + opacity only, written straight to the DOM in rAF.
 */
export function Story({ ids, cols, tile, width: W, height: H, photos, onPick }: Props) {
  const rows = Math.ceil(ids.length / cols);
  const step = tile + GAP;
  const stageH = rows * step;
  const D = 1.5 * H;  // scroll distance the stage stays pinned for
  const hr = Math.min(rows - 1, Math.floor(H / 2 / step)), hc = Math.floor((cols - 1) / 2);
  const hero = Math.min(ids.length - 1, hr * cols + hc);
  const hx = hc * step, hy = hr * step;

  const outer = useRef<HTMLDivElement>(null);
  const grid = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const img = useRef<HTMLImageElement>(null);
  const card = useRef<HTMLDivElement>(null);

  const lastP = useRef(-1);
  const update = () => {
    const o = outer.current, b = box.current, im = img.current, g = card.current?.firstElementChild as HTMLElement | null;
    if (!o || !b || !im || !g || REDUCED) return;
    const t = -o.getBoundingClientRect().top;  // px scrolled past the top of this item
    // open while it rises into view, hold while pinned, fold back before it scrolls on
    const p = ease(t < 0.25 * D ? clamp((t + 0.5 * H) / (0.25 * D + 0.5 * H)) : clamp((D - t) / (0.4 * D)));
    if (Math.abs(p - lastP.current) < 1e-4) return;  // holding (or fully closed): nothing to write this frame
    lastP.current = p;

    // FLIP: the box is laid out at its open size (W x H); scale it down onto the hero tile's slot, and counter-scale
    // the photo inside so it stays undistorted and covers the box (scale k), centred.
    const x = lerp(hx, 0, p), y = lerp(hy, 0, p), w = lerp(tile, W, p), h = lerp(tile, H, p);
    const sx = w / W, sy = h / H, k = Math.max(sx, sy);
    b.style.transform = `translate(${x}px, ${y}px) scale(${sx}, ${sy})`;
    im.style.transform = `translate(${(w - W * k) / (2 * sx)}px, ${(h - H * k) / (2 * sy)}px) scale(${k / sx}, ${k / sy})`;

    // the other tiles part around it and fade
    const kids = grid.current?.children;
    if (kids) for (let i = 0; i < kids.length; i++) {
      if (i === hero) continue;
      const e = kids[i] as HTMLElement, [dx, dy] = offsets[i];
      e.style.transform = p ? `translate3d(${dx * 0.3 * p}px, ${dy * 0.3 * p}px, 0) scale(${1 - 0.1 * p})` : "";
      e.style.opacity = String(1 - p);
    }

    // the glass panel settles in once the photo is mostly open
    const q = ease(clamp((p - 0.55) / 0.45));
    g.style.opacity = String(q);
    g.style.transform = `translateY(${(1 - q) * 28}px) scale(${0.97 + 0.03 * q})`;
    g.style.visibility = q > 0 ? "visible" : "hidden";  // no backdrop-filter work while invisible
  };

  // each tile's offset from the hero, computed once per layout instead of every frame
  const offsets = ids.map((_, i) => [(i % cols) * step - hc * step, Math.floor(i / cols) * step - hr * step]);

  useLayoutEffect(() => { lastP.current = -1; update(); }, [W, H, tile, cols, ids.length]);  // place before the first paint

  // fetch + decode the big hero photo well before it opens (a cold 1024px decode mid-animation was a ~200 ms stall)
  useEffect(() => {
    if (REDUCED) return;
    const pre = new Image();
    pre.src = api.thumb(ids[hero], 1024);
    pre.decode().catch(() => {});
  }, [ids, hero]);

  useEffect(() => {
    if (REDUCED || !outer.current) return;
    let raf = 0, on = false;
    const scroll = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; update(); }); };
    // only follow the scroll while the item is near the viewport
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting === on) return;
      on = e.isIntersecting;
      if (on) { window.addEventListener("scroll", scroll, { passive: true }); scroll(); }
      else window.removeEventListener("scroll", scroll);
    }, { rootMargin: "100% 0px" });
    io.observe(outer.current);
    return () => { io.disconnect(); window.removeEventListener("scroll", scroll); cancelAnimationFrame(raf); };
  }, [W, H, tile, cols, ids.length]);  // update() derives from these

  const tiles = (
    <div ref={grid} className="grid gap-1 pb-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {ids.map((id, i) => (i === hero ? (REDUCED ? null : <div key={id} className="aspect-square" />) : (
        <div key={id} className="origin-center will-change-transform"><Tile id={id} onPick={onPick} /></div>
      )))}
    </div>
  );

  if (REDUCED) {  // a static panel, then the stage's tiles as plain rows
    return (
      <>
        <section className="relative mb-1 flex items-center justify-center overflow-hidden px-5" style={{ height: Math.min(H * 0.85, 720) }}>
          <img src={api.thumb(ids[hero], 1024)} alt="" className="absolute inset-0 h-full w-full object-cover" />
          <Panel photos={photos} />
        </section>
        {tiles}
      </>
    );
  }

  return (
    <div ref={outer} style={{ height: stageH + D }}>
      <section className="sticky top-0 overflow-hidden" style={{ height: stageH }}>
        {tiles}
        <div ref={box} className="absolute left-0 top-0 overflow-hidden bg-black/5 will-change-transform dark:bg-white/5" style={{ width: W, height: H, transformOrigin: "0 0" }}>
          <img ref={img} src={api.thumb(ids[hero], 1024)} alt="" decoding="async" className="h-full w-full object-cover will-change-transform" style={{ transformOrigin: "0 0" }} />
        </div>
        <div ref={card} className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-center px-5 [&>*]:will-change-[transform,opacity]" style={{ height: H }}>
          <Panel photos={photos} />
        </div>
      </section>
    </div>
  );
}

function Panel({ photos }: { photos?: number }) {
  return (
    <Glass variant="regular" refract={false} radius={30} frost={14}
      className="w-full max-w-[660px] px-7 py-7 text-black/85 sm:px-10 sm:py-9 dark:text-white/90">
      <h2 className="text-balance text-[28px] font-semibold leading-tight tracking-tight sm:text-[38px]">Every photo, one sentence away.</h2>
      <ul className="mt-5 space-y-2.5 text-[16px] leading-snug sm:text-[19px]">
        <li>Search {photos ? photos.toLocaleString() : "thousands of"} photos in 10 ms, entirely on your device.</li>
        <li>Ask anything: dates, places, the words written in a photo.</li>
        <li>Find other photos of the same place from a single shot.</li>
        <li>Nothing leaves your device.</li>
      </ul>
      <div className="mt-6 text-[13px] font-medium tracking-wide text-black/50 dark:text-white/55">Just start typing.</div>
    </Glass>
  );
}
