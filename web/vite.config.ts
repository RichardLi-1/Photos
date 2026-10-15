import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Everything stays on this machine: the dev server and the Python API both bind to 127.0.0.1.
const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { host: "127.0.0.1", port: 5173, proxy: { "/api": api, "/photo": api } },
  preview: { host: "127.0.0.1" },
});
