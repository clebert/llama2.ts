import { computeArgmax } from './compute_argmax.js';
import { loadWasmModules } from './load_wasm_modules.js';
import { Decoder } from '@llama2/decoder';
import { createDataSource, loadCheckpoint, loadConfig, loadVocab } from '@llama2/loader';
import { Tokenizer } from '@llama2/tokenizer';
import { open } from 'node:fs/promises';
import { join } from 'node:path';
import { stdout } from 'node:process';

await loadWasmModules();

// const modelPath = `models/tinystories_110m`;
// const modelPath = `models/tinystories_42m`;
const modelPath = `models/tinystories_15m`;
// const modelPath = `models/tinystories_260k`;

const checkpointFile = await open(join(modelPath, `checkpoint_v1.bin`));

const checkpointDataSource = createDataSource(
  checkpointFile.readableWebStream() as ReadableStream<ArrayBuffer>,
);

await checkpointDataSource.next();

const config = await loadConfig(checkpointDataSource);

console.log({ config });

const sequenceLength = config.maxSequenceLength;
const checkpoint = await loadCheckpoint(checkpointDataSource, config, { sequenceLength });

await checkpointFile.close();

const tokenizerFile = await open(join(modelPath, `tokenizer.bin`));

const tokenizerDataSource = createDataSource(
  tokenizerFile.readableWebStream() as ReadableStream<ArrayBuffer>,
);

await tokenizerDataSource.next();

const vocab = await loadVocab(tokenizerDataSource, config.vocabSize);
const tokenizer = new Tokenizer(vocab);
const decoder = new Decoder(config, checkpoint);

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
