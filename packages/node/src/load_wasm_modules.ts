import { AdditionLayer, AttentionLayer, FnnLayer, LinearLayer } from '@llama2/decoder-wasm';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const wasmPath = join(dirname(fileURLToPath(import.meta.resolve(`@llama2/decoder-wasm`))), `wasm`);

export async function loadWasmModules(): Promise<void> {
  AdditionLayer.wasmModule = await loadWasmModule(`addition_layer`);
  AttentionLayer.wasmModule = await loadWasmModule(`attention_layer`);
  FnnLayer.wasmModule = await loadWasmModule(`fnn_layer`);
  LinearLayer.wasmModule = await loadWasmModule(`linear_layer`);
}

async function loadWasmModule(moduleName: string): Promise<WebAssembly.Module> {
  return WebAssembly.compile(await readFile(join(wasmPath, `${moduleName}.wasm`)));
}
