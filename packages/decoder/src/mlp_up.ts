import type { ModelConfig } from './create_checkpoint.js';

import { checkNotNull } from './check_not_null.js';

export class MlpUp {
  readonly normWeight: Uint8Array;
  readonly gateWeight: Uint8Array;
  readonly upWeight: Uint8Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  private readonly self: number;

  constructor(
    private readonly wasmInstance: WebAssembly.Instance,
    modelConfig: ModelConfig,
  ) {
    const exports = wasmInstance.exports as any as Exports;
    const { hiddenSize: inputSize, intermediateSize: outputSize, numLayers } = modelConfig;

    this.self = exports.create(inputSize, outputSize, numLayers);

    this.normWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getNormWeight(this.self), `normWeight`),
      numLayers * inputSize * 4,
    );

    this.gateWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getGateWeight(this.self), `gateWeight`),
      numLayers * outputSize * inputSize * 4,
    );

    this.upWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getUpWeight(this.self), `upWeight`),
      numLayers * outputSize * inputSize * 4,
    );

    this.inputVector = new Float32Array(
      exports.memory.buffer,
      checkNotNull(exports.getInputVector(this.self), `inputVector`),
      inputSize,
    );

    this.outputVector = new Float32Array(
      exports.memory.buffer,
      checkNotNull(exports.getOutputVector(this.self), `outputVector`),
      outputSize,
    );
  }

  forward(layer: number): void {
    (this.wasmInstance.exports as any as Exports).forward(this.self, layer);
  }
}

interface Exports {
  readonly memory: WebAssembly.Memory;

  create(inputSize: number, outputSize: number, numLayers: number): number;
  getNormWeight(self: number): number;
  getGateWeight(self: number): number;
  getUpWeight(self: number): number;
  getInputVector(self: number): number;
  getOutputVector(self: number): number;
  forward(self: number, layer: number): void;
}
