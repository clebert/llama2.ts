import type { ModelConfig } from './create_checkpoint.js';

import { checkNotNull } from './check_not_null.js';

export class Attention {
  readonly normWeight: Uint8Array;
  readonly queryWeight: Uint8Array;
  readonly keyWeight: Uint8Array;
  readonly valueWeight: Uint8Array;
  readonly outputWeight: Uint8Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  private readonly self: number;

  constructor(
    modelConfig: ModelConfig,
    private readonly wasmInstance: WebAssembly.Instance,
  ) {
    const exports = wasmInstance.exports as any as Exports;

    const {
      hiddenSize: querySize,
      maxSequenceLength,
      numLayers,
      numQueryHeads,
      numKeyValueHeads,
    } = modelConfig;

    this.self = exports.create(
      querySize,
      maxSequenceLength,
      numLayers,
      numQueryHeads,
      numKeyValueHeads,
    );

    this.normWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getNormWeight(this.self)),
      numLayers * querySize * 4,
    );

    this.queryWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getQueryWeight(this.self)),
      numLayers * querySize * querySize * 4,
    );

    const keyValueSize = numKeyValueHeads * (querySize / numQueryHeads);

    this.keyWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getKeyWeight(this.self)),
      numLayers * keyValueSize * querySize * 4,
    );

    this.valueWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getValueWeight(this.self)),
      numLayers * keyValueSize * querySize * 4,
    );

    this.outputWeight = new Uint8Array(
      exports.memory.buffer,
      checkNotNull(exports.getOutputWeight(this.self)),
      numLayers * querySize * querySize * 4,
    );

    this.inputVector = new Float32Array(
      exports.memory.buffer,
      checkNotNull(exports.getInputVector(this.self)),
      querySize,
    );

    this.outputVector = new Float32Array(
      exports.memory.buffer,
      checkNotNull(exports.getOutputVector(this.self)),
      querySize,
    );
  }

  forward(position: number, layer: number): void {
    (this.wasmInstance.exports as any as Exports).forward(this.self, position, layer);
  }
}

interface Exports {
  readonly memory: WebAssembly.Memory;

  create(
    querySize: number,
    maxSequenceLength: number,
    numLayers: number,
    numQueryHeads: number,
    numKeyValueHeads: number,
  ): number;

  getNormWeight(self: number): number;
  getQueryWeight(self: number): number;
  getKeyWeight(self: number): number;
  getValueWeight(self: number): number;
  getOutputWeight(self: number): number;
  getInputVector(self: number): number;
  getOutputVector(self: number): number;
  forward(self: number, position: number, layer: number): void;
}
