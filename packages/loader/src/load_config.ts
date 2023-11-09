import type { DataSource } from './create_data_source.js';
import type { Config } from '@llama2/decoder';

export async function loadConfig(dataSource: DataSource): Promise<Config> {
  const header = new Uint8Array(256);

  await dataSource.next(header);

  const headerView = new DataView(header.buffer);

  let byteOffset = 0;

  const magic = headerView.getUint32(byteOffset, true);
  const version = headerView.getUint32((byteOffset += 4), true);

  if (magic !== 0x616b3432) {
    throw new Error(`unknown checkpoint format`);
  }

  if (version !== 1) {
    throw new Error(`unsupported checkpoint version`);
  }

  return {
    embeddingSize: headerView.getInt32((byteOffset += 4), true),
    hiddenSize: headerView.getInt32((byteOffset += 4), true),
    layerCount: headerView.getInt32((byteOffset += 4), true),
    queryHeadCount: headerView.getInt32((byteOffset += 4), true),
    keyValueHeadCount: headerView.getInt32((byteOffset += 4), true),
    vocabSize: headerView.getInt32((byteOffset += 4), true),
    maxSequenceLength: headerView.getInt32((byteOffset += 4), true),
    sharedOutputWeight: headerView.getUint8((byteOffset += 4)) === 1,
  };
}
