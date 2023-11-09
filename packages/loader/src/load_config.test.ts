import type { Config } from '@llama2/decoder';

import { createDataSource } from './create_data_source.js';
import { loadConfig } from './load_config.js';
import { expect, test } from '@jest/globals';
import { open, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const fakeCheckpointPath = join(tmpdir(), `fake.bin`);

test(`tinystories 15m config`, async () => {
  const file = await open(`models/tinystories_15m/checkpoint_v1.bin`);
  const dataSource = createDataSource(file.readableWebStream() as ReadableStream<ArrayBuffer>);

  await dataSource.next();

  const config = await loadConfig(dataSource);

  await dataSource.next(); // close stream

  expect(config).toStrictEqual({
    embeddingSize: 288,
    hiddenSize: 768,
    layerCount: 6,
    queryHeadCount: 6,
    keyValueHeadCount: 6,
    vocabSize: 32000,
    maxSequenceLength: 256,
    sharedOutputWeight: true,
  } satisfies Config);
});

test(`tinystories 260k config`, async () => {
  const file = await open(`models/tinystories_260k/checkpoint_v1.bin`);
  const dataSource = createDataSource(file.readableWebStream() as ReadableStream<ArrayBuffer>);

  await dataSource.next();

  const config = await loadConfig(dataSource);

  await dataSource.next(); // close stream

  expect(config).toStrictEqual({
    embeddingSize: 64,
    hiddenSize: 172,
    layerCount: 5,
    queryHeadCount: 8,
    keyValueHeadCount: 4,
    vocabSize: 512,
    maxSequenceLength: 512,
    sharedOutputWeight: true,
  } satisfies Config);
});

test(`bad magic`, async () => {
  const buffer = Buffer.alloc(256);

  await writeFile(fakeCheckpointPath, buffer);

  const file = await open(fakeCheckpointPath);
  const dataSource = createDataSource(file.readableWebStream() as ReadableStream<ArrayBuffer>);

  await dataSource.next();

  await expect(loadConfig(dataSource)).rejects.toThrow(`unknown checkpoint format`);

  await dataSource.next(); // close stream
});

test(`bad version`, async () => {
  const buffer = Buffer.alloc(256);

  buffer.writeUInt32LE(0x616b3432);

  await writeFile(fakeCheckpointPath, buffer);

  const file = await open(fakeCheckpointPath);
  const dataSource = createDataSource(file.readableWebStream() as ReadableStream<ArrayBuffer>);

  await dataSource.next();

  await expect(loadConfig(dataSource)).rejects.toThrow(`unsupported checkpoint version`);

  await dataSource.next(); // close stream
});
