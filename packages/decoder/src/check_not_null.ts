export function checkNotNull(byteOffset: number, dataName: string): number {
  if (byteOffset <= 0) {
    throw new Error(`OOM: ${dataName}`);
  }

  return byteOffset;
}
