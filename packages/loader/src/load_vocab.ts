import type { DataSource } from './create_data_source.js';
import type { ModelConfig } from './load_model_config.js';

export interface Vocab {
  readonly entriesByToken: Map<string, VocabEntry>;
  readonly entriesByTokenId: VocabEntry[];
}

export interface VocabEntry {
  readonly score: number;
  readonly token: string;
  readonly tokenId: number;
}

export async function loadVocab(dataSource: DataSource, modelConfig: ModelConfig): Promise<Vocab> {
  const entriesByToken = new Map<string, VocabEntry>();
  const entriesByTokenId: VocabEntry[] = [];

  for (let tokenId = 0; tokenId < modelConfig.vocabSize; tokenId += 1) {
    const scoreData = new Float32Array(1);

    await dataSource.next(scoreData);

    const score = scoreData[0]!;
    const tokenByteLengthData = new Int32Array(1);

    await dataSource.next(tokenByteLengthData);

    const tokenByteLength = tokenByteLengthData[0]!;
    const tokenData = new Uint8Array(tokenByteLength);

    await dataSource.next(tokenData);

    const token = new TextDecoder(`utf-8`, { ignoreBOM: true })
      .decode(tokenData)
      .replaceAll(`▁`, ` `);

    const entry: VocabEntry = { score, token, tokenId };

    entriesByToken.set(token, entry);
    entriesByTokenId.push(entry);
  }

  return { entriesByToken, entriesByTokenId };
}
