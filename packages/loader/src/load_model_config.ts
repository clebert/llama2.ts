import type { DataSource } from './create_data_source.js';
import type { ModelConfig } from '@llama2/decoder';

export async function loadModelConfig(dataSource: DataSource): Promise<ModelConfig> {
  const configData = new Uint8Array(256);
  const configDataView = new DataView(configData.buffer);

  await dataSource.next(configData);

  const version = configData[0];

  let byteOffset = 1;

  const modelTypeByteLength = configDataView.getInt32(byteOffset, true);

  const modelType = new TextDecoder().decode(
    configData.subarray((byteOffset += 4), (byteOffset += modelTypeByteLength)),
  );

  if (version === 1 && modelType === `llama`) {
    return {
      version,
      modelType,
      hiddenSize: configDataView.getInt32(byteOffset, true),
      intermediateSize: configDataView.getInt32((byteOffset += 4), true),
      maxSequenceLength: configDataView.getInt32((byteOffset += 4), true),
      vocabSize: configDataView.getInt32((byteOffset += 4), true),
      numLayers: configDataView.getInt32((byteOffset += 4), true),
      numQueryHeads: configDataView.getInt32((byteOffset += 4), true),
      numKeyValueHeads: configDataView.getInt32((byteOffset += 4), true),
      sharedOutputWeight: configDataView.getUint8((byteOffset += 4)) === 1,
    };
  }

  throw new Error(`Unknown model config.`);
}
