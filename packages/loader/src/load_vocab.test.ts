import { createDataSource } from './create_data_source.js';
import { loadHyperparams } from './load_hyperparams.js';
import { loadVocab } from './load_vocab.js';
import { expect, test } from '@jest/globals';
import { open } from 'node:fs/promises';

test(`tinystories 15m vocab`, async () => {
  const file = await open(`models/tinystories_15m.bin`);
  const dataSource = createDataSource(file.readableWebStream() as ReadableStream<ArrayBuffer>);

  await dataSource.next();

  const hyperparams = await loadHyperparams(dataSource);

  const vocab = await loadVocab(dataSource, hyperparams.vocabSize);

  await dataSource.next(); // close stream

  expect(vocab.entriesByToken.size).toBe(32000);
  expect(vocab.entriesByTokenId.length).toBe(32000);

  const unkEntry = vocab.entriesByTokenId[0];

  expect(unkEntry).toStrictEqual({ score: 0, token: `<unk>`, tokenId: 0 });
  expect(vocab.entriesByToken.get(unkEntry!.token)).toBe(unkEntry);

  const utf8Entry = vocab.entriesByTokenId[31409]!;

  expect(utf8Entry).toStrictEqual({ score: -31150, token: `株`, tokenId: 31409 });
  expect(vocab.entriesByToken.get(utf8Entry!.token)).toBe(utf8Entry);

  const spaceEntry = vocab.entriesByTokenId[2913]!;

  expect(spaceEntry).toStrictEqual({ score: -2654, token: ` space`, tokenId: 2913 });
  expect(vocab.entriesByToken.get(spaceEntry!.token)).toBe(spaceEntry);
});
