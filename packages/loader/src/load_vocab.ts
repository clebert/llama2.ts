import type { DataSource } from './create_data_source.js';
import type { Header } from './load_header.js';

export interface Vocab {
  readonly entriesByToken: Map<string, VocabEntry>;
  readonly entriesByTokenId: VocabEntry[];
}

export interface VocabEntry {
  readonly tokenId: number;
  readonly token: string;
  readonly score: number;
}

export async function loadVocab(dataSource: DataSource, header: Header): Promise<Vocab> {
  const entriesByToken = new Map<string, VocabEntry>();
  const entriesByTokenId: VocabEntry[] = [];

  for (let tokenId = 0; tokenId < header.modelConfig.vocabSize; tokenId += 1) {
    const scoreData = new Float32Array(1);

    await dataSource.next(scoreData);

    const score = scoreData[0]!;
    const tokenLengthData = new Uint32Array(1);

    await dataSource.next(tokenLengthData);

    const tokenLength = tokenLengthData[0]!;
    const tokenData = new Uint8Array(tokenLength);

    await dataSource.next(tokenData);

    const token = new TextDecoder().decode(tokenData).replaceAll(`▁`, ` `);
    const entry: VocabEntry = { tokenId, token, score };

    entriesByToken.set(token, entry);
    entriesByTokenId.push(entry);
  }

  return { entriesByToken, entriesByTokenId };
}
