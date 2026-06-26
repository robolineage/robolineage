import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      // More-specific rules must come before less-specific ones (Vite matches in declaration order).
      '/api/health': { target: 'http://localhost:8081', changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') },
      '/api/session': { target: 'http://localhost:8080', changeOrigin: true, rewrite: (p) => p.replace(/^\/api\/session/, '') },
      '/stream': { target: 'http://localhost:8080', changeOrigin: true },
      '/mjpeg': { target: 'http://localhost:8080', changeOrigin: true },
    },
  },
})
