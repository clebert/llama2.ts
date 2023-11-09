import { AdditionLayer, AttentionLayer, FnnLayer, LinearLayer } from '@llama2/decoder';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const moduleUrl = import.meta.resolve(`@llama2/decoder`);
const libPath = join(fileURLToPath(moduleUrl), `../../zig-out/lib`);

export async function loadWasmModules(): Promise<void> {
  AdditionLayer.wasmModule = await loadWasmModule(join(libPath, `addition_layer.wasm`));
  AttentionLayer.wasmModule = await loadWasmModule(join(libPath, `attention_layer.wasm`));
  FnnLayer.wasmModule = await loadWasmModule(join(libPath, `fnn_layer.wasm`));
  LinearLayer.wasmModule = await loadWasmModule(join(libPath, `linear_layer.wasm`));
}

async function loadWasmModule(path: string): Promise<WebAssembly.Module> {
  return new WebAssembly.Module(await readFile(path));
}
