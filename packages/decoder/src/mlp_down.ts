import type { ModelConfig } from './create_checkpoint.js';

import { checkNotNull } from './check_not_null.js';

export class MlpDown {
  readonly downWeight: Uint8Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;
  readonly residualVector: Float32Array;

  private readonly self: number;

  constructor(
    modelConfig: ModelConfig,
    private readonly wasmInstance: WebAssembly.Instance,
  ) {
    const exports = wasmInstance.exports as any as Exports;
    const { hiddenSize: outputSize, intermediateSize: inputSize, numLayers } = modelConfig;

    this.self = exports.create(inputSize, outputSize, numLayers);

    this.downWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getDownWeight(this.self)),
      numLayers * outputSize * inputSize * 4,
    );

    this.inputVector = new Float32Array(
      exports.memory.buffer,
      checkNotNull(exports.getInputVector(this.self)),
      inputSize,
    );

    this.outputVector = new Float32Array(
      exports.memory.buffer,
      checkNotNull(exports.getOutputVector(this.self)),
      outputSize,
    );

    this.residualVector = new Float32Array(
      exports.memory.buffer,
      checkNotNull(exports.getResidualVector(this.self)),
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
  getDownWeight(self: number): number;
  getInputVector(self: number): number;
  getOutputVector(self: number): number;
  getResidualVector(self: number): number;
  forward(self: number, layer: number): void;
}
