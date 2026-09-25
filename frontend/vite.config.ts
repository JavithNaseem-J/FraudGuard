import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig(({ mode }) => {
  const envRoot = path.resolve(import.meta.dirname, '..');
  const env = loadEnv(mode, envRoot, '');
  const apiTarget = process.env.VITE_API_URL || env.VITE_API_URL || 'http://localhost:8000';

  return {
    envDir: envRoot,
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(import.meta.dirname, './src'),
      },
    },
    server: {
      port: 5173,
      host: true,
      proxy: {
        // Forward /api/* to FastAPI in development. Set VITE_API_URL in
        // .env.local when the backend runs on a non-default port.
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          rewrite: (proxyPath) => proxyPath.replace(/^\/api/, ''),
        },
      },
    },
  };
});
