import { createDataSource } from './create_data_source.js';
import { expect, test } from '@jest/globals';

const decoder = new TextDecoder();
const encoder = new TextEncoder();

function createTextStream(...chunks: readonly string[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller): void {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }

      controller.close();
    },
  });
}

test(`empty source data`, async () => {
  const dataSource = createDataSource(createTextStream());
  const targetChunk = encoder.encode(`.`);

  await dataSource.next();
  await dataSource.next();

  await expect(dataSource.next(targetChunk)).rejects.toThrow(`Unexpected end of stream.`);
});

test(`exhausted source data: s -> tt`, async () => {
  const dataSource = createDataSource(createTextStream(`a`));
  const targetChunk = encoder.encode(`..`);

  await dataSource.next();

  await expect(dataSource.next(targetChunk)).rejects.toThrow(`Unexpected end of stream.`);
  expect(decoder.decode(targetChunk)).toBe(`a.`);
});

test(`exhausted source data: ss -> t tt`, async () => {
  const dataSource = createDataSource(createTextStream(`ab`));
  const targetChunk1 = encoder.encode(`.`);
  const targetChunk2 = encoder.encode(`..`);

  await dataSource.next();
  await dataSource.next(targetChunk1);

  await expect(dataSource.next(targetChunk2)).rejects.toThrow(`Unexpected end of stream.`);
  expect(decoder.decode(targetChunk1)).toBe(`a`);
  expect(decoder.decode(targetChunk2)).toBe(`b.`);
});

test(`exhausted source data: s s -> ttt`, async () => {
  const dataSource = createDataSource(createTextStream(`a`, `b`));
  const targetChunk = encoder.encode(`...`);

  await dataSource.next();

  await expect(dataSource.next(targetChunk)).rejects.toThrow(`Unexpected end of stream.`);
  expect(decoder.decode(targetChunk)).toBe(`ab.`);
});

test(`exhausted source data: s s -> t tt`, async () => {
  const dataSource = createDataSource(createTextStream(`a`, `b`));
  const targetChunk1 = encoder.encode(`.`);
  const targetChunk2 = encoder.encode(`..`);

  await dataSource.next();
  await dataSource.next(targetChunk1);

  await expect(dataSource.next(targetChunk2)).rejects.toThrow(`Unexpected end of stream.`);
  expect(decoder.decode(targetChunk1)).toBe(`a`);
  expect(decoder.decode(targetChunk2)).toBe(`b.`);
});

test(`fully consumed source data: s -> t`, async () => {
  const dataSource = createDataSource(createTextStream(`a`));
  const targetChunk = encoder.encode(`.`);

  await dataSource.next();
  await dataSource.next(targetChunk);

  expect(decoder.decode(targetChunk)).toBe(`a`);
});

test(`fully consumed source data: ss -> t t`, async () => {
  const dataSource = createDataSource(createTextStream(`ab`));
  const targetChunk1 = encoder.encode(`.`);
  const targetChunk2 = encoder.encode(`.`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);

  expect(decoder.decode(targetChunk1)).toBe(`a`);
  expect(decoder.decode(targetChunk2)).toBe(`b`);
});

test(`fully consumed source data: sss -> t t t`, async () => {
  const dataSource = createDataSource(createTextStream(`abc`));
  const targetChunk1 = encoder.encode(`.`);
  const targetChunk2 = encoder.encode(`.`);
  const targetChunk3 = encoder.encode(`.`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);
  await dataSource.next(targetChunk3);

  expect(decoder.decode(targetChunk1)).toBe(`a`);
  expect(decoder.decode(targetChunk2)).toBe(`b`);
  expect(decoder.decode(targetChunk3)).toBe(`c`);
});

test(`fully consumed source data: s s -> tt`, async () => {
  const dataSource = createDataSource(createTextStream(`a`, `b`));
  const targetChunk = encoder.encode(`..`);

  await dataSource.next();
  await dataSource.next(targetChunk);

  expect(decoder.decode(targetChunk)).toBe(`ab`);
});

test(`fully consumed source data: s s -> t t`, async () => {
  const dataSource = createDataSource(createTextStream(`a`, `b`));
  const targetChunk1 = encoder.encode(`.`);
  const targetChunk2 = encoder.encode(`.`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);

  expect(decoder.decode(targetChunk1)).toBe(`a`);
  expect(decoder.decode(targetChunk2)).toBe(`b`);
});

test(`fully consumed source data: s ss -> tt t`, async () => {
  const dataSource = createDataSource(createTextStream(`a`, `bc`));
  const targetChunk1 = encoder.encode(`..`);
  const targetChunk2 = encoder.encode(`.`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);

  expect(decoder.decode(targetChunk1)).toBe(`ab`);
  expect(decoder.decode(targetChunk2)).toBe(`c`);
});

test(`fully consumed source data: s ss -> t t t`, async () => {
  const dataSource = createDataSource(createTextStream(`a`, `bc`));
  const targetChunk1 = encoder.encode(`.`);
  const targetChunk2 = encoder.encode(`.`);
  const targetChunk3 = encoder.encode(`.`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);
  await dataSource.next(targetChunk3);

  expect(decoder.decode(targetChunk1)).toBe(`a`);
  expect(decoder.decode(targetChunk2)).toBe(`b`);
  expect(decoder.decode(targetChunk3)).toBe(`c`);
});

test(`fully consumed source data: ss s -> t tt`, async () => {
  const dataSource = createDataSource(createTextStream(`ab`, `c`));
  const targetChunk1 = encoder.encode(`.`);
  const targetChunk2 = encoder.encode(`..`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);

  expect(decoder.decode(targetChunk1)).toBe(`a`);
  expect(decoder.decode(targetChunk2)).toBe(`bc`);
});

test(`fully consumed source data: ss s -> t t t`, async () => {
  const dataSource = createDataSource(createTextStream(`ab`, `c`));
  const targetChunk1 = encoder.encode(`.`);
  const targetChunk2 = encoder.encode(`.`);
  const targetChunk3 = encoder.encode(`.`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);
  await dataSource.next(targetChunk3);

  expect(decoder.decode(targetChunk1)).toBe(`a`);
  expect(decoder.decode(targetChunk2)).toBe(`b`);
  expect(decoder.decode(targetChunk3)).toBe(`c`);
});

test(`fully consumed source data: s s s -> ttt`, async () => {
  const dataSource = createDataSource(createTextStream(`a`, `b`, `c`));
  const targetChunk = encoder.encode(`...`);

  await dataSource.next();
  await dataSource.next(targetChunk);

  expect(decoder.decode(targetChunk)).toBe(`abc`);
});

test(`fully consumed source data: s s s -> t tt`, async () => {
  const dataSource = createDataSource(createTextStream(`a`, `b`, `c`));
  const targetChunk1 = encoder.encode(`.`);
  const targetChunk2 = encoder.encode(`..`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);

  expect(decoder.decode(targetChunk1)).toBe(`a`);
  expect(decoder.decode(targetChunk2)).toBe(`bc`);
});

test(`fully consumed source data: s s s -> tt t`, async () => {
  const dataSource = createDataSource(createTextStream(`a`, `b`, `c`));
  const targetChunk1 = encoder.encode(`..`);
  const targetChunk2 = encoder.encode(`.`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);

  expect(decoder.decode(targetChunk1)).toBe(`ab`);
  expect(decoder.decode(targetChunk2)).toBe(`c`);
});

test(`fully consumed source data: s s s -> t t t`, async () => {
  const dataSource = createDataSource(createTextStream(`a`, `b`, `c`));
  const targetChunk1 = encoder.encode(`.`);
  const targetChunk2 = encoder.encode(`.`);
  const targetChunk3 = encoder.encode(`.`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);
  await dataSource.next(targetChunk3);

  expect(decoder.decode(targetChunk1)).toBe(`a`);
  expect(decoder.decode(targetChunk2)).toBe(`b`);
  expect(decoder.decode(targetChunk3)).toBe(`c`);
});

test(`hidden source data`, async () => {
  const dataSource = createDataSource(
    new ReadableStream({
      start(controller): void {
        controller.enqueue(new Uint8Array(encoder.encode(`...abcdefghi...`).buffer, 3, 9));
        controller.enqueue(encoder.encode(`jkl`));
        controller.close();
      },
    }),
  );

  const targetChunk1 = encoder.encode(`......`);
  const targetChunk2 = encoder.encode(`......`);

  await dataSource.next();
  await dataSource.next(targetChunk1);
  await dataSource.next(targetChunk2);

  expect(decoder.decode(targetChunk1)).toBe(`abcdef`);
  expect(decoder.decode(targetChunk2)).toBe(`ghijkl`);
});

test(`hidden target data`, async () => {
  const dataSource = createDataSource(createTextStream(`abc`, `def`));
  const hiddenTargetChunk = encoder.encode(`123......456`);
  const targetChunk = new Uint8Array(hiddenTargetChunk.buffer, 3, 6);

  await dataSource.next();
  await dataSource.next(targetChunk);

  expect(decoder.decode(hiddenTargetChunk)).toBe(`123abcdef456`);
  expect(decoder.decode(targetChunk)).toBe(`abcdef`);
});
