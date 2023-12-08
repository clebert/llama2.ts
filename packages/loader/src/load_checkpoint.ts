import type { DataSource } from './create_data_source.js';
import type { ModelConfig } from './load_model_config.js';
import type { Checkpoint } from '@llama2/decoder';

import { Attention, Linear, MlpDown, MlpUp } from '@llama2/decoder';

export async function loadCheckpoint(
  dataSource: DataSource,
  modelConfig: ModelConfig,
): Promise<Checkpoint> {
  const {
    hiddenSize,
    intermediateSize,
    maxSequenceLength,
    vocabSize,
    numLayers,
    numQueryHeads,
    numKeyValueHeads,
    sharedOutputWeight,
  } = modelConfig;

  const embeddingWeight = new Uint8Array(vocabSize * hiddenSize * 4);

  await dataSource.next(embeddingWeight);

  const attention = await Attention.create({
    querySize: hiddenSize,
    maxSequenceLength,
    numLayers,
    numQueryHeads,
    numKeyValueHeads,
  });

  await dataSource.next(attention.normWeight);
  await dataSource.next(attention.queryWeight);
  await dataSource.next(attention.keyWeight);
  await dataSource.next(attention.valueWeight);
  await dataSource.next(attention.outputWeight);

  const mlpUp = await MlpUp.create({
    inputSize: hiddenSize,
    outputSize: intermediateSize,
    numLayers,
  });

  await dataSource.next(mlpUp.normWeight);
  await dataSource.next(mlpUp.gateWeight);
  await dataSource.next(mlpUp.upWeight);

  const mlpDown = await MlpDown.create({
    inputSize: intermediateSize,
    outputSize: hiddenSize,
    numLayers,
  });

  await dataSource.next(mlpDown.downWeight);

  const linear = await Linear.create({ inputSize: hiddenSize, outputSize: vocabSize });

  await dataSource.next(linear.normWeight);

  if (sharedOutputWeight) {
    linear.outputWeight.set(embeddingWeight);
  } else {
    await dataSource.next(linear.outputWeight);
  }

  return { embeddingWeight, attention, mlpUp, mlpDown, linear };
}
