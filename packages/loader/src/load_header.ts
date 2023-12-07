import type { DataSource } from './create_data_source.js';
import type { ModelConfig } from '@llama2/decoder';

export interface Header {
  readonly modelConfig: ModelConfig;
  readonly sharedOutputWeight: boolean;
}

export async function loadHeader(dataSource: DataSource): Promise<Header> {
  const headerData = new Uint8Array(256);
  const headerDataView = new DataView(headerData.buffer);

  await dataSource.next(headerData);

  const version = headerData[0];

  let byteOffset = 1;

  const modelTypeByteLength = headerDataView.getInt32(byteOffset, true);

  const modelType = new TextDecoder().decode(
    headerData.subarray((byteOffset += 4), (byteOffset += modelTypeByteLength)),
  );

  if (version === 1 && modelType === `llama`) {
    const modelConfig: ModelConfig = {
      hiddenSize: headerDataView.getInt32(byteOffset, true),
      intermediateSize: headerDataView.getInt32((byteOffset += 4), true),
      maxSequenceLength: headerDataView.getInt32((byteOffset += 4), true),
      vocabSize: headerDataView.getInt32((byteOffset += 4), true),
      numLayers: headerDataView.getInt32((byteOffset += 4), true),
      numQueryHeads: headerDataView.getInt32((byteOffset += 4), true),
      numKeyValueHeads: headerDataView.getInt32((byteOffset += 4), true),
    };

    const sharedOutputWeight = headerData[(byteOffset += 4)] === 1;

    return { modelConfig, sharedOutputWeight };
  }

  throw new Error(`unknown header`);
}
