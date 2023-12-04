export function checkNotNull(ptr: number): number {
  if (ptr <= 0) {
    throw new Error(`OOM`);
  }

  return ptr;
}
