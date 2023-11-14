import { complete } from './complete.js';
import { loadWasmModules } from './load_wasm_modules.js';
import yargs from 'yargs';

await loadWasmModules();

await yargs(process.argv.slice(2))
  .command(
    `complete <modelPath>`,
    ``,
    (args) =>
      args
        .positional(`modelPath`, { type: `string`, demandOption: true })
        .options({ prompt: { type: `string` }, maxSequenceLength: { type: `number` } }),
    async (args) => complete(args),
  )
  .demandCommand()
  .strict()
  .parseAsync();
