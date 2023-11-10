import type { DataSource } from './create_data_source.js';
import type { Hyperparams } from '@llama2/decoder';

export async function loadHyperparams(dataSource: DataSource): Promise<Hyperparams> {
  const headerData = new Uint8Array(256);

  await dataSource.next(headerData);

  const dataFormatMagic = new TextDecoder().decode(headerData.subarray(0, 6));

  if (dataFormatMagic !== `llama2`) {
    throw new Error(`Unknown data format.`);
  }

  const dataFormatVersion = headerData[6];

  if (dataFormatVersion !== 1) {
    throw new Error(`Unsupported data format version.`);
  }

  const headerView = new DataView(headerData.buffer);

  let byteOffset = 7;

  return {
    embeddingSize: headerView.getInt32(byteOffset, true),
    hiddenSize: headerView.getInt32((byteOffset += 4), true),
    keyValueSize: headerView.getInt32((byteOffset += 4), true),
    layerCount: headerView.getInt32((byteOffset += 4), true),
    queryHeadCount: headerView.getInt32((byteOffset += 4), true),
    vocabSize: headerView.getInt32((byteOffset += 4), true),
    maxSequenceLength: headerView.getInt32((byteOffset += 4), true),
    sharedOutputWeight: headerView.getUint8((byteOffset += 4)) === 1,
  };
}
