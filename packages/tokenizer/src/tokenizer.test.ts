import type { Vocab, VocabEntry } from '@llama2/loader';

import { Tokenizer } from './tokenizer.js';
import { beforeAll, expect, test } from '@jest/globals';
import { createDataSource, loadHyperparams, loadVocab } from '@llama2/loader';
import { open } from 'node:fs/promises';

let vocab15m: Vocab;
let vocab260k: Vocab;

beforeAll(async () => {
  const file15m = await open(`models/tinystories_15m.bin`);

  const dataSource15m = createDataSource(
    file15m.readableWebStream().getReader() as ReadableStreamDefaultReader,
  );

  await dataSource15m.next();

  const hyperparams15m = await loadHyperparams(dataSource15m);

  vocab15m = await loadVocab(dataSource15m, hyperparams15m.vocabSize);

  await dataSource15m.next(); // close stream

  const file260k = await open(`models/tinystories_260k.bin`);

  const dataSource260k = createDataSource(
    file260k.readableWebStream().getReader() as ReadableStreamDefaultReader,
  );

  await dataSource260k.next();

  const hyperparams260k = await loadHyperparams(dataSource260k);

  vocab260k = await loadVocab(dataSource260k, hyperparams260k.vocabSize);

  await dataSource260k.next(); // close stream
});

function createFakeVocab(...tokens: string[]): Vocab {
  const entriesByTokenId: VocabEntry[] = [];
  const entriesByToken = new Map<string, VocabEntry>();

  tokens.forEach((token, tokenId) => {
    const entry: VocabEntry = { tokenId, token, score: Math.random() };

    entriesByTokenId.push(entry);
    entriesByToken.set(entry.token, entry);
  });

  return { entriesByToken, entriesByTokenId };
}

test(`unsupported vocab`, () => {
  expect(() => new Tokenizer(createFakeVocab(`foo`))).toThrow(
    `Unsupported vocab detected. Expected '<unk>' at position 0 but found 'foo' instead.`,
  );

  expect(() => new Tokenizer(createFakeVocab(`<unk>`, `foo`))).toThrow(
    `Unsupported vocab detected. Expected '<s>' at position 1 but found 'foo' instead.`,
  );

  expect(() => new Tokenizer(createFakeVocab(`<unk>`, `<s>`, `foo`))).toThrow(
    `Unsupported vocab detected. Expected '</s>' at position 2 but found 'foo' instead.`,
  );

  expect(() => new Tokenizer(createFakeVocab(`<unk>`, `<s>`, `</s>`, `foo`))).toThrow(
    `Unsupported vocab detected. Expected '<0x00>' at position 3 but found 'foo' instead.`,
  );

  expect(() => new Tokenizer(createFakeVocab(`<unk>`, `<s>`, `</s>`, `<0x00>`, `foo`))).toThrow(
    `Unsupported vocab detected. Expected '<0x01>' at position 4 but found 'foo' instead.`,
  );
});

test(`decode tokens`, () => {
  const tokenizer = new Tokenizer(vocab15m);

  expect(tokenizer.decode(450)).toBe(` The`);
  expect(tokenizer.decode(450, tokenizer.bosTokenId)).toBe(`The`);
  expect(tokenizer.decode(450, tokenizer.eosTokenId)).toBe(` The`);
  expect(tokenizer.decode(450, 467)).toBe(` The`);
});

test(`decode special tokens`, () => {
  const tokenizer = new Tokenizer(vocab260k);

  expect(tokenizer.unkTokenId).toBe(0);
  expect(tokenizer.bosTokenId).toBe(1);
  expect(tokenizer.eosTokenId).toBe(2);
  expect(tokenizer.decode(tokenizer.unkTokenId)).toBe(undefined);
  expect(tokenizer.decode(tokenizer.bosTokenId)).toBe(undefined);
  expect(tokenizer.decode(tokenizer.eosTokenId)).toBe(undefined);
});

test(`decode printable characters`, () => {
  const tokenizer = new Tokenizer(vocab15m);

  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x1F>`)!.tokenId)).toBe(`<0x1F>`);
  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x20>`)!.tokenId)).toBe(` `);
  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x21>`)!.tokenId)).toBe(`!`);
  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x7D>`)!.tokenId)).toBe(`}`);
  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x7E>`)!.tokenId)).toBe(`~`);
  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x7F>`)!.tokenId)).toBe(`<0x7F>`);
});

test(`decode whitespace`, () => {
  const tokenizer = new Tokenizer(vocab15m);

  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x09>`)!.tokenId)).toBe(`\t`);
  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x0A>`)!.tokenId)).toBe(`\n`);
  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x0B>`)!.tokenId)).toBe(`\v`);
  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x0C>`)!.tokenId)).toBe(`\f`);
  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x0D>`)!.tokenId)).toBe(`\r`);
  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0x20>`)!.tokenId)).toBe(` `);

  expect(tokenizer.decode(vocab15m.entriesByToken.get(`<0xA0>`)!.tokenId)).toBe(
    String.fromCharCode(0xa0),
  );
});

test(`encode utf-8`, () => {
  const tokenizer = new Tokenizer(vocab15m);
  const input = `Lets try ö & 株式会社`;
  const tokenIds = tokenizer.encode(input);

  expect(tokenIds).toStrictEqual([365, 1691, 1018, 3963, 669, 29871, 31409, 30607, 30437, 30564]);

  const output = tokenIds.map((tokenId) => vocab15m.entriesByTokenId[tokenId]!.token).join(``);

  expect(output).toBe(` ${input}`);
});

test(`encode empty string`, () => {
  const tokenizer = new Tokenizer(vocab15m);
  const input = ``;
  const tokenIds = tokenizer.encode(input);

  expect(tokenIds).toStrictEqual([]);
});

test(`encode blank string`, () => {
  const tokenizer = new Tokenizer(vocab15m);
  const input = ` \n `;
  const tokenIds = tokenizer.encode(input);

  expect(tokenIds).toStrictEqual([]);
});

test(`encode unknown tokens`, () => {
  const tokenizer = new Tokenizer(vocab15m);
  const input = `𒎗𓐍`;
  const tokenIds = tokenizer.encode(input);

  expect(tokenIds).toStrictEqual([29871, 243, 149, 145, 154, 243, 150, 147, 144]);

  const output = tokenIds.map((tokenId) => vocab15m.entriesByTokenId[tokenId]!.token).join(``);

  expect(output).toBe(` <0xF0><0x92><0x8E><0x97><0xF0><0x93><0x90><0x8D>`);
});

test(`encode single chars`, () => {
  const tokenizer = new Tokenizer(vocab260k);
  const input = `abcdefgh`;
  const tokenIds = tokenizer.encode(input);

  expect(tokenIds).toStrictEqual([261, 430, 429, 418, 411, 431, 428, 415]);

  const output = tokenIds.map((tokenId) => vocab260k.entriesByTokenId[tokenId]!.token).join(``);

  expect(output).toBe(` ${input}`);
});

// https://github.com/facebookresearch/llama/blob/ea9f33d6d3ea8ed7d560d270986407fd6c2e52b7/example_text_completion.py
test(`meta encoding example 1`, () => {
  const tokenizer = new Tokenizer(vocab15m);
  const input = `I believe the meaning of life is`;
  const tokenIds = tokenizer.encode(input);

  expect(tokenIds).toStrictEqual([306, 4658, 278, 6593, 310, 2834, 338]);

  const output = tokenIds.map((tokenId) => vocab15m.entriesByTokenId[tokenId]!.token).join(``);

  expect(output).toBe(` ${input}`);
});

test(`meta encoding example 2`, () => {
  const tokenizer = new Tokenizer(vocab15m);
  const input = `Simply put, the theory of relativity states that `;
  const tokenIds = tokenizer.encode(input);

  expect(tokenIds).toStrictEqual([
    3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871,
  ]);

  const output = tokenIds.map((tokenId) => vocab15m.entriesByTokenId[tokenId]!.token).join(``);

  expect(output).toBe(` ${input}`);
});

test(`meta encoding example 3`, () => {
  const tokenizer = new Tokenizer(vocab15m);
  const input = `A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just `;
  const tokenIds = tokenizer.encode(input);

  expect(tokenIds).toStrictEqual([
    319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13, 13, 4706, 6324,
    14332, 29892, 13, 13, 4706, 306, 925, 29871,
  ]);

  const output = tokenIds.map((tokenId) => vocab15m.entriesByTokenId[tokenId]!.token).join(``);

  expect(output).toBe(` ${input.replaceAll(`\n`, `<0x0A>`)}`);
});

test(`meta encoding example 4`, () => {
  const tokenizer = new Tokenizer(vocab15m);
  const input = `Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>`;
  const tokenIds = tokenizer.encode(input);

  expect(tokenIds).toStrictEqual([
    4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449, 276, 316,
    2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715,
    1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706, 923, 968, 1149,
  ]);

  const output = tokenIds.map((tokenId) => vocab15m.entriesByTokenId[tokenId]!.token).join(``);

  expect(output).toBe(` ${input.replaceAll(`\n`, `<0x0A>`)}`);
});
