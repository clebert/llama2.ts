import { AdditionLayer, AttentionLayer, FnnLayer, LinearLayer } from '@llama2/decoder';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const moduleUrl = import.meta.resolve(`@llama2/decoder`);

export async function loadWasmModules(): Promise<void> {
  AdditionLayer.wasmModule = await loadWasmModule(`addition_layer`);
  AttentionLayer.wasmModule = await loadWasmModule(`attention_layer`);
  FnnLayer.wasmModule = await loadWasmModule(`fnn_layer`);
  LinearLayer.wasmModule = await loadWasmModule(`linear_layer`);
}

async function loadWasmModule(moduleName: string): Promise<WebAssembly.Module> {
  return WebAssembly.compile(
    await readFile(join(fileURLToPath(moduleUrl), `../../zig-out/lib/${moduleName}.wasm`)),
  );
}
