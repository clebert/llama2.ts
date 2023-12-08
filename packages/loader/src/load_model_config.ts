import type { DataSource } from './create_data_source.js';

export interface ModelConfig {
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly maxSequenceLength: number;
  readonly vocabSize: number;
  readonly numLayers: number;
  readonly numQueryHeads: number;
  readonly numKeyValueHeads: number;
  readonly sharedOutputWeight: boolean;
}

export async function loadModelConfig(dataSource: DataSource): Promise<ModelConfig> {
  const modelConfigData = new Uint8Array(256);
  const modelConfigDataView = new DataView(modelConfigData.buffer);

  await dataSource.next(modelConfigData);

  const version = modelConfigData[0];

  let byteOffset = 1;

  const modelTypeByteLength = modelConfigDataView.getInt32(byteOffset, true);

  const modelType = new TextDecoder().decode(
    modelConfigData.subarray((byteOffset += 4), (byteOffset += modelTypeByteLength)),
  );

  if (version === 1 && modelType === `llama`) {
    return {
      hiddenSize: modelConfigDataView.getInt32(byteOffset, true),
      intermediateSize: modelConfigDataView.getInt32((byteOffset += 4), true),
      maxSequenceLength: modelConfigDataView.getInt32((byteOffset += 4), true),
      vocabSize: modelConfigDataView.getInt32((byteOffset += 4), true),
      numLayers: modelConfigDataView.getInt32((byteOffset += 4), true),
      numQueryHeads: modelConfigDataView.getInt32((byteOffset += 4), true),
      numKeyValueHeads: modelConfigDataView.getInt32((byteOffset += 4), true),
      sharedOutputWeight: modelConfigData[(byteOffset += 4)] === 1,
    };
  }

  throw new Error(`unknown model`);
}
