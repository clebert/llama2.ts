export type DataSource = AsyncGenerator<undefined, never, ArrayBufferView>;

export async function* createDataSource(
  stream: ReadableStream<ArrayBuffer | Uint8Array>,
): DataSource {
  const reader = stream.getReader();

  let targetChunk: Uint8Array | undefined;

  while (true) {
    const result = await reader.read();

    if (result.done) {
      break;
    }

    let sourceChunk: Uint8Array | undefined =
      result.value instanceof Uint8Array ? result.value : new Uint8Array(result.value);

    do {
      if (!targetChunk) {
        const nextTargetChunk = yield;

        if (!nextTargetChunk) {
          await reader.cancel();

          break;
        }

        targetChunk = new Uint8Array(
          nextTargetChunk.buffer,
          nextTargetChunk.byteOffset,
          nextTargetChunk.byteLength,
        );
      }

      if (targetChunk.byteLength >= sourceChunk.byteLength) {
        targetChunk.set(sourceChunk);

        targetChunk =
          targetChunk.byteLength > sourceChunk.byteLength
            ? new Uint8Array(
                targetChunk.buffer,
                targetChunk.byteOffset + sourceChunk.byteLength,
                targetChunk.byteLength - sourceChunk.byteLength,
              )
            : undefined;

        sourceChunk = undefined;
      } else {
        targetChunk.set(
          new Uint8Array(sourceChunk.buffer, sourceChunk.byteOffset, targetChunk.byteLength),
        );

        sourceChunk = new Uint8Array(
          sourceChunk.buffer,
          sourceChunk.byteOffset + targetChunk.byteLength,
          sourceChunk.byteLength - targetChunk.byteLength,
        );

        targetChunk = undefined;
      }
    } while (sourceChunk !== undefined);
  }

  while (true) {
    if (targetChunk || (yield)) {
      throw new Error(`unexpected end of stream`);
    }
  }
}
