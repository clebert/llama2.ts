import type { Hyperparams } from '@llama2/decoder';

import { createDataSource } from './create_data_source.js';
import { loadHyperparams } from './load_hyperparams.js';
import { expect, test } from '@jest/globals';
import { open, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const fakeCheckpointPath = join(tmpdir(), `fake.bin`);

test(`tinystories 15m hyperparams`, async () => {
  const file = await open(`models/tinystories_15m.bin`);
  const dataSource = createDataSource(file.readableWebStream() as ReadableStream<ArrayBuffer>);

  await dataSource.next();

  const hyperparams = await loadHyperparams(dataSource);

  await dataSource.next(); // close stream

  expect(hyperparams).toStrictEqual({
    embeddingSize: 288,
    hiddenSize: 768,
    keyValueSize: 288,
    layerCount: 6,
    queryHeadCount: 6,
    vocabSize: 32000,
    maxSequenceLength: 256,
    sharedOutputWeight: true,
  } satisfies Hyperparams);
});

test(`tinystories 260k hyperparams`, async () => {
  const file = await open(`models/tinystories_260k.bin`);
  const dataSource = createDataSource(file.readableWebStream() as ReadableStream<ArrayBuffer>);

  await dataSource.next();

  const hyperparams = await loadHyperparams(dataSource);

  await dataSource.next(); // close stream

  expect(hyperparams).toStrictEqual({
    embeddingSize: 64,
    hiddenSize: 172,
    keyValueSize: 32,
    layerCount: 5,
    queryHeadCount: 8,
    vocabSize: 512,
    maxSequenceLength: 512,
    sharedOutputWeight: true,
  } satisfies Hyperparams);
});

test(`unknown data format`, async () => {
  const buffer = Buffer.alloc(256);

  await writeFile(fakeCheckpointPath, buffer);

  const file = await open(fakeCheckpointPath);
  const dataSource = createDataSource(file.readableWebStream() as ReadableStream<ArrayBuffer>);

  await dataSource.next();

  await expect(loadHyperparams(dataSource)).rejects.toThrow(`Unknown data format.`);

  await dataSource.next(); // close stream
});

test(`unsupported data format version`, async () => {
  const buffer = Buffer.alloc(256);

  buffer.write(`llama2`);

  await writeFile(fakeCheckpointPath, buffer);

  const file = await open(fakeCheckpointPath);
  const dataSource = createDataSource(file.readableWebStream() as ReadableStream<ArrayBuffer>);

  await dataSource.next();

  await expect(loadHyperparams(dataSource)).rejects.toThrow(`Unsupported data format version.`);

  await dataSource.next(); // close stream
});
