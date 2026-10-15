import { Fragment, useEffect, useId, useLayoutEffect, useRef, useState, type HTMLAttributes } from "react";

/**
 * Liquid glass, physically based (Chromium):
 *  - the element is a slab of glass whose rim is a convex squircle lens;
 *  - per pixel, light through that surface is bent with Snell's law (n = 1.5) and the resulting
 *    offset is baked into a displacement map (SVG feDisplacementMap on the backdrop);
 *  - the rim stays crisp (you see the photos bend), the centre is frosted so text reads;
 *  - a specular map adds the light catching the rim (top-left key light + faint far-side bounce).
 * Safari/Firefox can't run SVG filters in backdrop-filter: they get frosted glass + the same specular rim.
 */
const chromium = typeof navigator !== "undefined" &&
  !!(navigator as Navigator & { userAgentData?: { brands: { brand: string }[] } }).userAgentData?.brands?.some((b) => /Chromium/.test(b.brand));

type Props = HTMLAttributes<HTMLDivElement> & {
  radius?: number;  // px; default = pill
  bezel?: number;   // px width of the curved rim
  depth?: number;   // rim bend, px
  zoom?: number;    // centre magnification of the lens (1 = none)
  pillow?: number;  // sideways pull near the top/bottom rims: straight lines bow into curves
  frost?: number;   // centre blur, px
  ior?: number;     // index of refraction
  variant?: "clear" | "regular";  // regular = Apple's denser glass for small text (cards, menus)
  refract?: boolean;  // false = frosted only: no displacement maps or SVG filter (cheap for big/animated panels)
};

export function Glass({ radius, bezel = 999, depth = 26, zoom = 1.02, pillow = 0.035, frost = 1.8, ior = 1.5, variant = "clear", refract: wantRefract = true, className = "", style, children, ...rest }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const id = `lg${useId().replace(/:/g, "")}`;
  const size = useSize(ref);
  const r = radius ?? (size ? size.h / 2 : 999);
  const maps = useGlassMaps(wantRefract ? size : null, r, bezel, ior, depth, zoom, pillow);  // skip the pixel loop when off

  const refract = chromium && maps && !NO_GLASS;
  const backdrop = refract ? `url(#${id})` : `blur(${frost * 2.5}px) saturate(180%)`;

  return (
    <div ref={ref} className={`liquid-glass ${variant === "regular" ? "liquid-glass-regular" : ""} ${className}`}
      style={{ borderRadius: r, backdropFilter: backdrop, WebkitBackdropFilter: backdrop, ...style }} {...rest}>
      {refract && (
        <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden>
          {/* filter region in % of the element; the displacement map below in px */}
          <filter id={id} x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
            {/* ONE feImage only: Chromium breaks feDisplacementMap when a backdrop filter has two. R/G = offset */}
            {/* map in exact px at (0,0): with width="100%" Chromium misplaces it and everything shifts (the "offset bars") */}
            <feImage href={maps.disp} x="0" y="0" width={maps.w} height={maps.h} preserveAspectRatio="none" result="map" />
            {/* R/G/B refract by slightly different amounts: a hint of dispersion at the rim */}
            {(["R", "G", "B"] as const).map((c, i) => (
              <Fragment key={c}>{/* no <g>: filter primitives must be direct children of <filter> */}
                <feDisplacementMap in="SourceGraphic" in2="map" scale={2 * maps.range * [1, 0.99, 0.98][i]}
                  xChannelSelector="R" yChannelSelector="G" result={`d${c}`} />
                <feColorMatrix in={`d${c}`} type="matrix" result={`c${c}`} values={CHANNEL[c]} />
              </Fragment>
            ))}
            <feBlend in="cR" in2="cG" mode="screen" result="rg" />
            <feBlend in="rg" in2="cB" mode="screen" result="refr" />
            {/* one uniform frost over the whole refracted image: no sharp-rim/frosted-centre seams (they read as offset bars) */}
            <feGaussianBlur in="refr" stdDeviation={frost} result="glass" />
            <feColorMatrix in="glass" type="saturate" values="1.35" />
          </filter>
        </svg>
      )}
      {maps && (
        <span aria-hidden className="pointer-events-none absolute inset-0 rounded-[inherit] mix-blend-screen"
          style={{ backgroundImage: `url(${maps.spec})`, backgroundSize: "100% 100%" }} />
      )}
      {children}
    </div>
  );
}

// ?noglass in the URL = frosted fallback only (for side-by-side comparison)
const NO_GLASS = typeof location !== "undefined" && new URLSearchParams(location.search).has("noglass");

const CHANNEL = {
  R: "1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0",
  G: "0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0",
  B: "0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0",
};

function useSize(ref: React.RefObject<HTMLDivElement | null>) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => {
      const b = e.borderBoxSize?.[0];
      setSize({ w: Math.round(b?.inlineSize ?? e.contentRect.width), h: Math.round(b?.blockSize ?? e.contentRect.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

/** Bake displacement (Snell refraction through a squircle rim), rim mask and specular highlight into images. */
function useGlassMaps(size: { w: number; h: number } | null, radius: number, bezel: number, ior: number, depth: number, zoom: number, pillow: number) {
  const [maps, setMaps] = useState<{ disp: string; spec: string; range: number; w: number; h: number } | null>(null);
  useEffect(() => {
    if (!size || size.w < 4 || size.h < 4) return;
    const W = size.w, H = size.h;
    const r = Math.min(radius, H / 2, W / 2), b = Math.min(bezel, H / 2, W / 2);

    // Refraction profile across the rim, t = 0 at the edge .. 1 where the glass turns flat.
    // The rim is a convex circular bevel h(t) = sqrt(1 - (1 - t)^2) (bezel 999 => the whole pill is a glass rod).
    // A vertical ray hitting a surface tilted by theta bends to theta_r = asin(sin(theta) / n); its lateral shift is
    // proportional to tan(theta - theta_r). Tilt is capped at 80 deg so the edge singularity can't eat the range.
    const N = 256, prof = new Float32Array(N + 1);
    let max = 0;
    for (let i = 0; i <= N; i++) {
      const t = i / N, u = 1 - t;
      const theta = Math.min(Math.atan(u / Math.sqrt(Math.max(1e-6, 1 - u * u))), (80 * Math.PI) / 180);
      const shift = Math.tan(theta - Math.asin(Math.sin(theta) / ior));
      prof[i] = shift; max = Math.max(max, shift);
    }
    for (let i = 0; i <= N; i++) prof[i] /= max || 1;

    const mk = () => { const c = document.createElement("canvas"); c.width = W; c.height = H; return c; };
    const cd = mk(), cs = mk();
    const xd = cd.getContext("2d")!, xs = cs.getContext("2d")!;
    const D = xd.createImageData(W, H), S = xs.createImageData(W, H);
    const RIM = new Uint8ClampedArray(W * H);
    const lx = 0.6, ly = 0.8; // light travelling down-right = key light from the top-left
    // Where each pixel samples the backdrop from (px): rim refraction pulls inward along the normal, and the
    // lens as a whole magnifies about its centre. Encoded into R/G relative to the largest offset ("range").
    const OX = new Float32Array(W * H), OY = new Float32Array(W * H);
    let range = 1;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const px = x + 0.5 - W / 2, py = y + 0.5 - H / 2;
        const qx = Math.abs(px) - (W / 2 - r), qy = Math.abs(py) - (H / 2 - r);
        let nx = 0, ny = 0, d: number;
        if (qx > 0 && qy > 0) { const L = Math.hypot(qx, qy); d = L - r; nx = (qx / L) * Math.sign(px); ny = (qy / L) * Math.sign(py); }
        else if (qx > qy) { d = qx - r; nx = Math.sign(px); }
        else { d = qy - r; ny = Math.sign(py); }
        const i = (y * W + x) * 4;
        const inside = d <= 0;
        const t = inside ? Math.min(1, -d / b) : 0;
        const m = inside ? prof[Math.round(t * N)] : 0;

        if (inside) {
          const k = 1 - 1 / zoom;
          // keep every sample inside the slab: the rim shows the glass's own content squeezed and wrapped
          const bow = pillow * Math.pow(Math.abs(py) / (H / 2), 2.5);  // 0 on the centre line -> pillow at the rim
          const sx = Math.max(-W / 2 + 1, Math.min(W / 2 - 1, px - nx * m * depth - px * k - px * bow));
          const sy = Math.max(-H / 2 + 1, Math.min(H / 2 - 1, py - ny * m * depth - py * k));
          OX[i / 4] = sx - px;
          OY[i / 4] = sy - py;
          range = Math.max(range, Math.abs(OX[i / 4]), Math.abs(OY[i / 4]));
        }

        const rim = inside ? Math.pow(1 - t, 1.5) : 0;            // 1 at the edge -> 0 in the flat centre
        RIM[i / 4] = Math.round(Math.min(1, rim * 1.6) * 255);

        const facing = -(nx * lx + ny * ly);                       // >0: rim faces the light
        const lit = Math.pow(Math.max(0, facing), 2) + 0.5 * Math.pow(Math.max(0, -facing), 3);
        const edge = inside ? Math.pow(Math.max(0, 1 - (-d) / (b * 0.6)), 1.8) : 0;
        S.data[i] = S.data[i + 1] = S.data[i + 2] = 255; S.data[i + 3] = Math.round(Math.min(1, lit * edge * 1.3) * 255);
      }
    }
    for (let j = 0; j < W * H; j++) {
      D.data[j * 4] = Math.round(127.5 + (OX[j] / range) * 127);
      D.data[j * 4 + 1] = Math.round(127.5 + (OY[j] / range) * 127);
      D.data[j * 4 + 2] = RIM[j]; D.data[j * 4 + 3] = 255;
    }
    xd.putImageData(D, 0, 0); xs.putImageData(S, 0, 0);
    setMaps({ disp: cd.toDataURL(), spec: cs.toDataURL(), range, w: W, h: H });
  }, [size?.w, size?.h, radius, bezel, ior, depth, zoom, pillow]);
  return maps;
}
