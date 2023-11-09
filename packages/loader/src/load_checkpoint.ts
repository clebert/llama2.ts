import type { DataSource } from './create_data_source.js';
import type { Checkpoint, Config } from '@llama2/decoder';

import { AttentionLayer, FnnLayer, LinearLayer } from '@llama2/decoder';

export interface LoadCheckpointOptions {
  readonly sequenceLength?: number;
}

export async function loadCheckpoint(
  dataSource: DataSource,
  config: Config,
  options?: LoadCheckpointOptions,
): Promise<Checkpoint> {
  const attentionLayers: AttentionLayer[] = [];
  const fnnLayers: FnnLayer[] = [];

  for (let index = 0; index < config.layerCount; index += 1) {
    attentionLayers.push(
      new AttentionLayer({
        inputVectorLength: config.embeddingSize,
        queryHeadCount: config.queryHeadCount,
        keyValueHeadCount: config.keyValueHeadCount,
        sequenceLength: options?.sequenceLength ?? config.maxSequenceLength,
      }),
    );

    fnnLayers.push(
      new FnnLayer({
        inputVectorLength: config.embeddingSize,
        hiddenVectorLength: config.hiddenSize,
      }),
    );
  }

  for (const attentionLayer of attentionLayers) {
    await dataSource.next(attentionLayer.normWeightVector);
  }

  for (const fnnLayer of fnnLayers) {
    await dataSource.next(fnnLayer.normWeightVector);
  }

  const linearLayer = new LinearLayer({
    inputVectorLength: config.embeddingSize,
    outputVectorLength: config.vocabSize,
  });

  await dataSource.next(linearLayer.normWeightVector);

  const embeddingVectors: Float32Array[] = [];

  for (let index = 0; index < config.vocabSize; index += 1) {
    const embeddingVector = new Float32Array(config.embeddingSize);

    await dataSource.next(embeddingVector);

    embeddingVectors.push(embeddingVector);
  }

  for (const attentionLayer of attentionLayers) {
    await dataSource.next(attentionLayer.queryWeightMatrix);
  }

  for (const attentionLayer of attentionLayers) {
    await dataSource.next(attentionLayer.keyWeightMatrix);
  }

  for (const attentionLayer of attentionLayers) {
    await dataSource.next(attentionLayer.valueWeightMatrix);
  }

  for (const attentionLayer of attentionLayers) {
    await dataSource.next(attentionLayer.outputWeightMatrix);
  }

  for (const fnnLayer of fnnLayers) {
    await dataSource.next(fnnLayer.gateWeightMatrix);
  }

  for (const fnnLayer of fnnLayers) {
    await dataSource.next(fnnLayer.downWeightMatrix);
  }

  for (const fnnLayer of fnnLayers) {
    await dataSource.next(fnnLayer.upWeightMatrix);
  }

  if (config.sharedOutputWeight) {
    for (let index = 0; index < embeddingVectors.length; index += 1) {
      const embeddingVector = embeddingVectors[index]!;

      linearLayer.outputWeightMatrix.set(embeddingVector, index * embeddingVector.length);
    }
  } else {
    await dataSource.next(linearLayer.outputWeightMatrix);
  }

  return { embeddingVectors, attentionLayers, fnnLayers, linearLayer };
}
