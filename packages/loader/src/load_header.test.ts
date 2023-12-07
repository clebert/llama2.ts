import type { ModelConfig } from '@llama2/decoder';

import { createDataSource } from './create_data_source.js';
import { loadHeader } from './load_header.js';
import { expect, test } from '@jest/globals';
import { open } from 'node:fs/promises';

test(`tinystories 15m header v1`, async () => {
  const file = await open(`models/tinystories_15m_v1.bin`);

  const dataSource = createDataSource(
    file.readableWebStream().getReader() as ReadableStreamDefaultReader,
  );

  await dataSource.next();

  const header = await loadHeader(dataSource);

  await dataSource.next(); // close stream

  expect(header).toStrictEqual({
    modelConfig: {
      hiddenSize: 288,
      intermediateSize: 768,
      maxSequenceLength: 256,
      vocabSize: 32000,
      numLayers: 6,
      numQueryHeads: 6,
      numKeyValueHeads: 6,
    } satisfies ModelConfig,

    sharedOutputWeight: true,
  });
});

test(`tinystories 260k header v1`, async () => {
  const file = await open(`models/tinystories_260k_v1.bin`);

  const dataSource = createDataSource(
    file.readableWebStream().getReader() as ReadableStreamDefaultReader,
  );

  await dataSource.next();

  const header = await loadHeader(dataSource);

  await dataSource.next(); // close stream

  expect(header).toStrictEqual({
    modelConfig: {
      hiddenSize: 64,
      intermediateSize: 172,
      maxSequenceLength: 512,
      vocabSize: 512,
      numLayers: 5,
      numQueryHeads: 8,
      numKeyValueHeads: 4,
    } satisfies ModelConfig,

    sharedOutputWeight: true,
  });
});
