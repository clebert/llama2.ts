import { computeArgmax } from './compute_argmax.js';
import { loadWasmModules } from './load_wasm_modules.js';
import { Decoder } from '@llama2/decoder';
import { createDataSource, loadCheckpoint, loadHyperparams, loadVocab } from '@llama2/loader';
import { Tokenizer } from '@llama2/tokenizer';
import { open } from 'node:fs/promises';
import { stdout } from 'node:process';
import yargs from 'yargs';

await loadWasmModules();

var argv = yargs(process.argv.slice(2))
  .options({
    modelPath: { type: `string`, demandOption: true },
    prompt: { type: `string`, default: `` },
    sequenceLength: { type: `number` },
  })
  .strict()
  .parseSync();

const file = await open(argv.modelPath);

const dataSource = createDataSource(
  file.readableWebStream().getReader() as ReadableStreamDefaultReader,
);

await dataSource.next();

const hyperparams = await loadHyperparams(dataSource);
const vocab = await loadVocab(dataSource, hyperparams.vocabSize);
const sequenceLength = argv.sequenceLength || hyperparams.maxSequenceLength;
const checkpoint = await loadCheckpoint(dataSource, hyperparams, { sequenceLength });

await file.close();

const tokenizer = new Tokenizer(vocab);
const decoder = await Decoder.instantiate(hyperparams, checkpoint);
const promptTokenIds = tokenizer.encode(argv.prompt);

let nextTokenId = promptTokenIds.shift() ?? tokenizer.bosTokenId;

const firstToken = tokenizer.decode(nextTokenId, tokenizer.bosTokenId);

if (firstToken) {
  stdout.write(firstToken);
}

for (let position = 0; position < sequenceLength; position += 1) {
  const tokenId = nextTokenId;
  const logits = decoder.decode(tokenId, position);

  nextTokenId = promptTokenIds.shift() ?? computeArgmax(logits);

  const nextToken = tokenizer.decode(nextTokenId, tokenId);

  if (nextToken === undefined) {
    break;
  }

  stdout.write(nextToken);
}

stdout.write(`\n`);
