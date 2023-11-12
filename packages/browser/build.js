// @ts-check

import * as esbuild from 'esbuild';
import { htmlPlugin } from 'esbuild-html-plugin';
import { rm } from 'node:fs/promises';
import process from 'node:process';

const outdir = `dist`;
const nodeEnv = process.env[`NODE_ENV`] ?? `production`;
const dev = nodeEnv !== `production`;

/** @type {import('esbuild').BuildOptions} */
const mainBuildOptions = {
  alias: {
    '@llama2/decoder': `../decoder/src/mod.ts`,
    '@llama2/loader': `../loader/src/mod.ts`,
    '@llama2/tokenizer': `../tokenizer/src/mod.ts`,
  },
  entryPoints: [
    { out: `main`, in: `src/main.ts` },
    { out: `worker`, in: `src/worker.ts` },
  ],
  entryNames: `[dir]/[name]-[hash]`,
  bundle: true,
  minify: !dev,
  sourcemap: dev,
  target: `es2022`,
  tsconfig: `tsconfig.json`,
  outdir,
  publicPath: `/static`,
  plugins: [
    htmlPlugin({
      outfile: `index.html`,
      language: `en`,

      createHeadElements: () => [
        `<meta charset="utf-8" />`,
        `<meta name="viewport" content="width=device-width, initial-scale=1" />`,
        `<title>llama2.ts</title>`,
      ],

      createBodyElements: (outputUrls) => {
        const mainScriptUrl = outputUrls.find((url) => url.includes(`main`) && url.endsWith(`.js`));

        const workerScriptUrl = outputUrls.find(
          (url) => url.includes(`worker`) && url.endsWith(`.js`),
        );

        return [
          `<script src="${mainScriptUrl}" async></script>`,
          `<script>workerScriptUrl = "${workerScriptUrl}";</script>`,
        ];
      },
    }),
  ],
};

if (process.argv.includes(`--watch`)) {
  await Promise.all([esbuild.context(mainBuildOptions).then((ctx) => ctx.watch())]);
} else {
  await rm(outdir, { recursive: true, force: true });

  await Promise.all([esbuild.build(mainBuildOptions)]);
}
