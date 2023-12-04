import type { DataSource } from './create_data_source.js';
import type { Checkpoint, ModelConfig } from '@llama2/decoder';

export async function loadCheckpoint(
  dataSource: DataSource,
  modelConfig: ModelConfig,
  checkpoint: Checkpoint,
): Promise<void> {
  const { sharedOutputWeight } = modelConfig;
  const { embeddingWeight, attention, mlpUp, mlpDown, linear } = checkpoint;

  await dataSource.next(checkpoint.embeddingWeight);

  await dataSource.next(attention.normWeight);
  await dataSource.next(attention.queryWeight);
  await dataSource.next(attention.keyWeight);
  await dataSource.next(attention.valueWeight);
  await dataSource.next(attention.outputWeight);

  await dataSource.next(mlpUp.normWeight);
  await dataSource.next(mlpUp.gateWeight);
  await dataSource.next(mlpUp.upWeight);
  await dataSource.next(mlpDown.downWeight);

  await dataSource.next(linear.normWeight);

  if (sharedOutputWeight) {
    linear.outputWeight.set(embeddingWeight);
  } else {
    await dataSource.next(linear.outputWeight);
  }
}
