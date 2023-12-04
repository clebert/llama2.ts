import { complete } from './complete.js';
import yargs from 'yargs';

await yargs(process.argv.slice(2))
  .command(
    `complete <modelPath>`,
    ``,
    (args) =>
      args
        .positional(`modelPath`, { type: `string`, demandOption: true })
        .options({ prompt: { type: `string` }, maxSequenceLength: { type: `number` } }),
    async (args) => {
      try {
        await complete(args);
      } catch (error) {
        console.error(error);
      }
    },
  )
  .demandCommand()
  .strict()
  .parseAsync();
