/// <reference types="vitest/config" />
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server: /api is proxied to the Zyqora backend (local, or the Raspberry Pi):
//   ZYQORA_API_TARGET=http://<pi-ip>:8000 npm run dev
const apiTarget = process.env.ZYQORA_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: { port: 5173, proxy: { "/api": { target: apiTarget, changeOrigin: true } } },
  build: {
    target: "es2022",
    // The three.js chunk (~885 kB, ~240 kB gzip) is lazy-loaded only for the optional Overview illustration.
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router"],
          charts: ["recharts"],
          three: ["three", "@react-three/fiber"],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
