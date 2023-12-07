import { Attention } from './attention.js';
import { Linear } from './linear.js';
import { MlpDown } from './mlp_down.js';
import { MlpUp } from './mlp_up.js';

export interface WasmModules {
  readonly attention: WebAssembly.Module;
  readonly mlpUp: WebAssembly.Module;
  readonly mlpDown: WebAssembly.Module;
  readonly linear: WebAssembly.Module;
}

export interface ModelConfig {
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly maxSequenceLength: number;
  readonly vocabSize: number;
  readonly numLayers: number;
  readonly numQueryHeads: number;
  readonly numKeyValueHeads: number;
}

export interface Checkpoint {
  readonly embeddingWeight: Uint8Array;
  readonly attention: Attention;
  readonly mlpUp: MlpUp;
  readonly mlpDown: MlpDown;
  readonly linear: Linear;
}

export async function createCheckpoint(
  modelConfig: ModelConfig,
  wasmModules: WasmModules,
): Promise<Checkpoint> {
  return {
    embeddingWeight: new Uint8Array(modelConfig.vocabSize * modelConfig.hiddenSize * 4),
    attention: new Attention(modelConfig, await WebAssembly.instantiate(wasmModules.attention)),
    mlpUp: new MlpUp(modelConfig, await WebAssembly.instantiate(wasmModules.mlpUp)),
    mlpDown: new MlpDown(modelConfig, await WebAssembly.instantiate(wasmModules.mlpDown)),
    linear: new Linear(modelConfig, await WebAssembly.instantiate(wasmModules.linear)),
  };
}
