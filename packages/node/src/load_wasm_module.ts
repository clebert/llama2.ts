import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const wasmPath = join(dirname(fileURLToPath(import.meta.resolve(`@llama2/decoder`))), `wasm`);

export async function loadWasmModule(moduleName: string): Promise<WebAssembly.Module> {
  return WebAssembly.compile(await readFile(join(wasmPath, `${moduleName}.wasm`)));
}
