// @ts-check

import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const wasmPath = join(dirname(fileURLToPath(import.meta.resolve(`@llama2/decoder`))), `wasm`);

/** @type {import('aws-simple').ConfigFileDefaultExport} */
export default () => ({
  terminationProtectionEnabled: true,
  routes: [
    {
      type: `file`,
      publicPath: `/*`,
      path: `dist/index.html`,
      responseHeaders: { 'cache-control': `no-store` },
    },
    {
      type: `folder`,
      publicPath: `/static/*`,
      path: `dist`,
      responseHeaders: { 'cache-control': `max-age=157680000` }, // 5 years
    },
    {
      type: `folder`,
      publicPath: `/static/models/*`,
      path: `../../models`,
      responseHeaders: { 'cache-control': `max-age=157680000` }, // 5 years
    },
    {
      type: `folder`,
      publicPath: `/static/wasm/*`,
      path: wasmPath,
      responseHeaders: { 'cache-control': `max-age=157680000` }, // 5 years
    },
  ],
});
