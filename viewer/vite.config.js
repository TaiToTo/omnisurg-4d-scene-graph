// Build the viewer as a static page, and serve a local data directory at
// `data/` while developing.
//
// `VIEWER_DATA=/path/to/data npm run dev` serves that directory, the one
// `python -m pipeline.viewer_catalog` wrote, next to the page. A built page
// reads its data from `data/` beside it, or from `VITE_DATA_ROOT` when that is
// set at build time.
import fs from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'vite';

const MIME = {
  '.json': 'application/json',
  '.glb': 'model/gltf-binary',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
};

/** Serve `root` under `/data/`, refusing any path that leaves it. */
function serveData(root) {
  const base = path.resolve(root);
  return (req, res, next) => {
    let rel;
    try {
      const { pathname } = new URL(req.url, 'http://localhost');
      if (!pathname.startsWith('/data/')) return next();
      rel = decodeURIComponent(pathname.slice('/data/'.length));
    } catch {
      res.statusCode = 400;
      res.end('Bad Request');
      return undefined;
    }
    const file = path.resolve(base, rel);
    if (!file.startsWith(base + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      res.statusCode = 404;
      res.end('Not found');
      return undefined;
    }
    res.setHeader('Content-Type', MIME[path.extname(file).toLowerCase()] ?? 'application/octet-stream');
    // Regenerated data must not be served from the browser's cache.
    res.setHeader('Cache-Control', 'no-store');
    fs.createReadStream(file).pipe(res);
    return undefined;
  };
}

const dataPlugin = {
  name: 'serve-viewer-data',
  configureServer(server) {
    if (process.env.VIEWER_DATA) server.middlewares.use(serveData(process.env.VIEWER_DATA));
  },
  configurePreviewServer(server) {
    if (process.env.VIEWER_DATA) server.middlewares.use(serveData(process.env.VIEWER_DATA));
  },
};

export default defineConfig({
  // Relative asset paths, so the built page works from any directory of a host.
  base: './',
  publicDir: false,
  // three.js alone is about 600 kB minified; one chunk for one page is fine.
  build: { chunkSizeWarningLimit: 800 },
  plugins: [dataPlugin],
  server: { port: Number(process.env.VIEWER_PORT) || 5180 },
  test: {
    environment: 'node',
    include: ['src/**/*.test.js'],
  },
});
