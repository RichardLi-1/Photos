// The hosted demo's public address. Everywhere else this UI runs (Vite dev, the local search/server.py, a `docker run`
// of the demo, any other deploy) is a test build. The native Mac, iOS and Android apps don't load this UI at all.
const PRODUCTION_HOSTS = ["photos-demo.fly.dev"];

/** "Test build" pill, same look as bpat's. Top-left, since the search bar owns the bottom; never takes clicks. */
export function TestBadge() {
  if (PRODUCTION_HOSTS.includes(location.hostname)) return null;
  return (
    <span className="pointer-events-none fixed top-[max(16px,env(safe-area-inset-top))] left-[max(16px,env(safe-area-inset-left))] z-50 rounded-full border border-amber-500/40 bg-amber-100/95 px-3.5 py-2 text-[0.6rem] font-bold uppercase tracking-[0.2em] text-amber-800 shadow-[0_10px_30px_-10px_rgba(23,55,58,0.45)]">
      Test build
    </span>
  );
}
