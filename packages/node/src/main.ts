import { computeArgmax } from './compute_argmax.js';
import { loadWasmModules } from './load_wasm_modules.js';
import { Decoder } from '@llama2/decoder';
import { createDataSource, loadCheckpoint, loadHyperparams, loadVocab } from '@llama2/loader';
import { Tokenizer } from '@llama2/tokenizer';
import { open } from 'node:fs/promises';
import { stdout } from 'node:process';

await loadWasmModules();

// const modelPath = `../../models/tinystories_110m.bin`;
// const modelPath = `../../models/tinystories_42m.bin`;
const modelPath = `../../models/tinystories_15m.bin`;
// const modelPath = `../../models/tinystories_260k.bin`;

const file = await open(modelPath);

const dataSource = createDataSource(
  file.readableWebStream().getReader() as ReadableStreamDefaultReader,
);

await dataSource.next();

const hyperparams = await loadHyperparams(dataSource);

console.log({ hyperparams });

const vocab = await loadVocab(dataSource, hyperparams.vocabSize);
const sequenceLength = hyperparams.maxSequenceLength;
const checkpoint = await loadCheckpoint(dataSource, hyperparams, { sequenceLength });

await file.close();

const tokenizer = new Tokenizer(vocab);
const decoder = await Decoder.instantiate(hyperparams, checkpoint);

const tokenIds = [tokenizer.bosTokenId];

let totalTime = 0;

while (tokenIds.length <= sequenceLength) {
  const position = tokenIds.length - 1;
  const tokenId = tokenIds[position]!;

  let startTime = 0;

  if (position > 0) {
    startTime = performance.now();
  }

  const logits = decoder.decode(tokenId, position);

  if (startTime > 0) {
    totalTime += performance.now() - startTime;
  }

  const nextTokenId = computeArgmax(logits);
  const nextToken = tokenizer.decode(nextTokenId, tokenId);

  if (nextToken === undefined) {
    break;
  }

  stdout.write(nextToken);
  tokenIds.push(nextTokenId);
}

if (tokenIds.length > 1) {
  const averageTime = totalTime / (tokenIds.length - 1);

  console.log(`\n\nachieved: ${(1000 / averageTime).toFixed(3)} tok/s`);
}
