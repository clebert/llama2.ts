import type { ModelConfig } from '@llama2/decoder';

import { createDataSource } from './create_data_source.js';
import { loadModelConfig } from './load_model_config.js';
import { expect, test } from '@jest/globals';
import { open } from 'node:fs/promises';

test(`tinystories 15m model config`, async () => {
  const file = await open(`models/tinystories_15m_v1.bin`);

  const dataSource = createDataSource(
    file.readableWebStream().getReader() as ReadableStreamDefaultReader,
  );

  await dataSource.next();

  const modelConfig = await loadModelConfig(dataSource);

  await dataSource.next(); // close stream

  expect(modelConfig).toStrictEqual({
    version: 1,
    modelType: `llama`,
    hiddenSize: 288,
    intermediateSize: 768,
    maxSequenceLength: 256,
    vocabSize: 32000,
    numLayers: 6,
    numQueryHeads: 6,
    numKeyValueHeads: 6,
    sharedOutputWeight: true,
  } satisfies ModelConfig);
});

test(`tinystories 260k model config`, async () => {
  const file = await open(`models/tinystories_260k_v1.bin`);

  const dataSource = createDataSource(
    file.readableWebStream().getReader() as ReadableStreamDefaultReader,
  );

  await dataSource.next();

  const modelConfig = await loadModelConfig(dataSource);

  await dataSource.next(); // close stream

  expect(modelConfig).toStrictEqual({
    version: 1,
    modelType: `llama`,
    hiddenSize: 64,
    intermediateSize: 172,
    maxSequenceLength: 512,
    vocabSize: 512,
    numLayers: 5,
    numQueryHeads: 8,
    numKeyValueHeads: 4,
    sharedOutputWeight: true,
  } satisfies ModelConfig);
});
