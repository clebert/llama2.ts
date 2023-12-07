import type { ModelConfig } from './create_checkpoint.js';

import { checkNotNull } from './check_not_null.js';

export class Linear {
  readonly normWeight: Uint8Array;
  readonly outputWeight: Uint8Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  private readonly self: number;

  constructor(
    modelConfig: ModelConfig,
    private readonly wasmInstance: WebAssembly.Instance,
  ) {
    const exports = wasmInstance.exports as any as Exports;
    const { hiddenSize, vocabSize } = modelConfig;

    this.self = exports.create(hiddenSize, vocabSize);

    this.normWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getNormWeight(this.self)),
      hiddenSize * 4,
    );

    this.outputWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getOutputWeight(this.self)),
      vocabSize * hiddenSize * 4,
    );

    this.inputVector = new Float32Array(
      exports.memory.buffer,
      checkNotNull(exports.getInputVector(this.self)),
      hiddenSize,
    );

    this.outputVector = new Float32Array(
      exports.memory.buffer,
      checkNotNull(exports.getOutputVector(this.self)),
      vocabSize,
    );
  }

  forward(computeSoftmax: boolean): void {
    (this.wasmInstance.exports as any as Exports).forward(this.self, computeSoftmax);
  }
}

interface Exports {
  readonly memory: WebAssembly.Memory;

  create(inputSize: number, outputSize: number): number;
  getNormWeight(self: number): number;
  getOutputWeight(self: number): number;
  getInputVector(self: number): number;
  getOutputVector(self: number): number;
  forward(self: number, computeSoftmax: boolean): void;
}
