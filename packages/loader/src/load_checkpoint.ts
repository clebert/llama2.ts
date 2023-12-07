import type { DataSource } from './create_data_source.js';
import type { Header } from './load_header.js';
import type { Checkpoint } from '@llama2/decoder';

export async function loadCheckpoint(
  dataSource: DataSource,
  header: Header,
  checkpoint: Checkpoint,
): Promise<void> {
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

  if (header.sharedOutputWeight) {
    linear.outputWeight.set(embeddingWeight);
  } else {
    await dataSource.next(linear.outputWeight);
  }
}
