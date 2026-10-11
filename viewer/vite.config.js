// Build the viewer as a static page, and serve the bundle directory named by VIEWER_DATA at ./data/ in
// development and preview. A deployment puts the bundle at ./data/ beside the built page, or names its
// URL at build time with VITE_DATA_URL.
import fs from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'vite';

const TYPES = { '.json': 'application/json', '.glb': 'model/gltf-binary', '.jpg': 'image/jpeg' };

/**
 * Serve directory `dataDir` read-only at /data/: GET and HEAD of the bundle's file types, under the directory
 * after symbolic links are resolved. A request is refused unless it names localhost, an IP address or an
 * allowed host, as Vite refuses one for its own files.
 */
export function bundleDirectory(dataDir = process.env.VIEWER_DATA) {
  const root = dataDir ? fs.realpathSync(path.resolve(dataDir)) : null;
  const inside = (file) => {
    const rel = path.relative(root, file);
    return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
  };
  const LOCAL = /^(localhost|[\w-]+\.localhost|\d{1,3}(\.\d{1,3}){3}|\[[0-9a-f:.]+\])(:\d+)?$/i;
  const refuse = (res, code) => { res.statusCode = code; res.end(); };
  const serve = (allowedHosts) => (req, res, next) => {
    const url = new URL(req.url, 'http://localhost');
    if (!root || !url.pathname.startsWith('/data/')) return next();
    const host = req.headers.host ?? '';
    if (!LOCAL.test(host) && allowedHosts !== true && !(allowedHosts ?? []).includes(host.replace(/:\d+$/, ''))) {
      return refuse(res, 403);
    }
    let file, stat;
    try {
      file = fs.realpathSync(path.resolve(root, decodeURIComponent(url.pathname.slice('/data/'.length))));
      stat = fs.statSync(file);
    } catch {
      return refuse(res, 404);
    }
    const type = TYPES[path.extname(file)];
    if (!['GET', 'HEAD'].includes(req.method) || !type || !inside(file) || !stat.isFile()) return refuse(res, 404);
    res.setHeader('Content-Type', type);
    res.setHeader('Content-Length', stat.size);
    if (req.method === 'HEAD') return res.end();
    // A file that cannot be read, or that a new bundle replaces while it is sent, ends this response only;
    // an unhandled stream error would stop the server.
    const stream = fs.createReadStream(file);
    stream.on('error', () => {
      if (res.headersSent) { res.destroy(); return; }
      res.removeHeader('Content-Length');
      refuse(res, 404);
    });
    stream.pipe(res);
    return undefined;
  };
  return {
    name: 'bundle-directory',
    configureServer(server) {
      if (!root && !process.env.VITEST) server.config.logger.warn('VIEWER_DATA is not set: the page will find no catalog.');
      server.middlewares.use(serve(server.config.server.allowedHosts));
    },
    configurePreviewServer(server) { server.middlewares.use(serve(server.config.preview.allowedHosts)); },
  };
}

export default defineConfig({
  // Relative URLs throughout, so the built page works from any directory of a static host.
  base: './',
  publicDir: false,
  server: { port: 5180 },
  preview: { port: 5181 },
  plugins: [bundleDirectory()],
  // three is most of the page's code and changes least, so it is a file of its own that a browser keeps.
  build: { rollupOptions: { output: { manualChunks: { three: ['three'] } } }, chunkSizeWarningLimit: 700 },
  test: { environment: 'node', include: ['src/**/*.test.js'] },
});
