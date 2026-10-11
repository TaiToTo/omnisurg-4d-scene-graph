// Serve a bundle directory through the development server's middleware over HTTP, and check what it refuses.
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { bundleDirectory } from '../vite.config.js';

let dir, server, port;

/** Ask the server for `url`, naming `host`, and return the status, the headers and the body. */
function ask(url, { method = 'GET', host = `localhost:${port}` } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({ port, path: url, method, headers: { host } }, (res) => {
      let body = '';
      res.on('data', (c) => { body += c; });
      res.on('end', () => resolve({ status: res.statusCode, type: res.headers['content-type'], body }));
    });
    req.on('error', reject);
    req.end();
  });
}

beforeAll(async () => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bundle-'));
  const data = path.join(dir, 'data');
  fs.mkdirSync(path.join(data, 'clip'), { recursive: true });
  fs.writeFileSync(path.join(data, 'catalog.json'), '{"format":1}');
  fs.writeFileSync(path.join(data, 'notes.txt'), 'not a bundle file');
  fs.writeFileSync(path.join(dir, 'secret.json'), '{"secret":1}');
  fs.symlinkSync(path.join(dir, 'secret.json'), path.join(data, 'clip', 'link.json'));
  fs.writeFileSync(path.join(data, 'clip', 'locked.json'), '{}');
  fs.chmodSync(path.join(data, 'clip', 'locked.json'), 0o000);

  let middleware;
  bundleDirectory(data).configureServer({
    config: { server: { allowedHosts: ['viewer.example'] }, logger: { warn() {} } },
    middlewares: { use: (fn) => { middleware = fn; } },
  });
  server = http.createServer((req, res) => middleware(req, res, () => { res.statusCode = 299; res.end('next'); }));
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  port = server.address().port;
});

afterAll(() => {
  server.close();
  fs.chmodSync(path.join(dir, 'data', 'clip', 'locked.json'), 0o600);
  fs.rmSync(dir, { recursive: true, force: true });
});

describe('the development server', () => {
  it('serves a bundle file with its type, and HEAD without a body', async () => {
    expect(await ask('/data/catalog.json')).toEqual({ status: 200, type: 'application/json', body: '{"format":1}' });
    expect(await ask('/data/catalog.json', { method: 'HEAD' })).toMatchObject({ status: 200, body: '' });
  });

  it('passes on what is not under /data/', async () => {
    expect((await ask('/index.html')).status).toBe(299);
  });

  it('serves localhost, an IP address and an allowed host only', async () => {
    for (const host of [`127.0.0.1:${port}`, `192.168.1.20:${port}`, '[::1]', 'viewer.example']) {
      expect((await ask('/data/catalog.json', { host })).status).toBe(200);
    }
    for (const host of ['evil.example', 'localhost.evil.example', 'notlocalhost', 'x127.0.0.1', '192.168.1.20.evil.example']) {
      expect((await ask('/data/catalog.json', { host })).status).toBe(403);
    }
  });

  it('refuses a write, a path or link out of the directory, a directory and another type', async () => {
    expect((await ask('/data/catalog.json', { method: 'POST' })).status).toBe(404);
    expect((await ask('/data/%2e%2e%2fsecret.json')).status).toBe(404);
    expect((await ask('/data/clip/link.json')).status).toBe(404);
    expect((await ask('/data/clip')).status).toBe(404);
    expect((await ask('/data/notes.txt')).status).toBe(404);
    expect((await ask('/data/missing.json')).status).toBe(404);
  });

  it('answers a file it cannot read with an error, and goes on serving', async () => {
    expect((await ask('/data/clip/locked.json')).status).toBe(404);
    expect((await ask('/data/catalog.json')).status).toBe(200);
  });
});
