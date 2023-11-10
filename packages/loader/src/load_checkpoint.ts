import type { DataSource } from './create_data_source.js';
import type { Checkpoint, Hyperparams } from '@llama2/decoder';

import { AttentionLayer, FnnLayer, LinearLayer } from '@llama2/decoder';

export interface LoadCheckpointOptions {
  readonly sequenceLength?: number;
}

export async function loadCheckpoint(
  dataSource: DataSource,
  hyperparams: Hyperparams,
  options?: LoadCheckpointOptions,
): Promise<Checkpoint> {
  const embeddingVectors: Float32Array[] = [];

  for (let index = 0; index < hyperparams.vocabSize; index += 1) {
    const embeddingVector = new Float32Array(hyperparams.embeddingSize);

    await dataSource.next(embeddingVector);

    embeddingVectors.push(embeddingVector);
  }

  const sequenceLength = options?.sequenceLength ?? hyperparams.maxSequenceLength;
  const attentionLayers: AttentionLayer[] = [];

  for (let index = 0; index < hyperparams.layerCount; index += 1) {
    const attentionLayer = new AttentionLayer({ ...hyperparams, sequenceLength });

    await dataSource.next(attentionLayer.normWeightVector);
    await dataSource.next(attentionLayer.queryWeightMatrix);
    await dataSource.next(attentionLayer.keyWeightMatrix);
    await dataSource.next(attentionLayer.valueWeightMatrix);
    await dataSource.next(attentionLayer.outputWeightMatrix);

    attentionLayers.push(attentionLayer);
  }

  const fnnLayers: FnnLayer[] = [];

  for (let index = 0; index < hyperparams.layerCount; index += 1) {
    const fnnLayer = new FnnLayer(hyperparams);

    await dataSource.next(fnnLayer.normWeightVector);
    await dataSource.next(fnnLayer.gateWeightMatrix);
    await dataSource.next(fnnLayer.upWeightMatrix);
    await dataSource.next(fnnLayer.downWeightMatrix);

    fnnLayers.push(fnnLayer);
  }

  const linearLayer = new LinearLayer(hyperparams);

  await dataSource.next(linearLayer.normWeightVector);

  if (hyperparams.sharedOutputWeight) {
    for (let index = 0; index < embeddingVectors.length; index += 1) {
      const embeddingVector = embeddingVectors[index]!;

      linearLayer.outputWeightMatrix.set(embeddingVector, index * embeddingVector.length);
    }
  } else {
    await dataSource.next(linearLayer.outputWeightMatrix);
  }

  return { embeddingVectors, attentionLayers, fnnLayers, linearLayer };
}
