import { checkNotNull } from './check_not_null.js';

export interface AttentionInit {
  readonly querySize: number;
  readonly maxSequenceLength: number;
  readonly numLayers: number;
  readonly numQueryHeads: number;
  readonly numKeyValueHeads: number;
}

export class Attention {
  static wasmModule: WebAssembly.Module;

  static async create(init: AttentionInit): Promise<Attention> {
    return new Attention(init, await WebAssembly.instantiate(this.wasmModule));
  }

  readonly normWeight: Uint8Array;
  readonly queryWeight: Uint8Array;
  readonly keyWeight: Uint8Array;
  readonly valueWeight: Uint8Array;
  readonly outputWeight: Uint8Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  private readonly self: number;

  private constructor(
    { querySize, maxSequenceLength, numLayers, numQueryHeads, numKeyValueHeads }: AttentionInit,
    private readonly wasmInstance: WebAssembly.Instance,
  ) {
    const {
      memory,
      create,
      getNormWeight,
      getQueryWeight,
      getKeyWeight,
      getValueWeight,
      getOutputWeight,
      getInputVector,
      getOutputVector,
    } = wasmInstance.exports as WasmExports;

    this.self = create(querySize, maxSequenceLength, numLayers, numQueryHeads, numKeyValueHeads);

    this.normWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getNormWeight(this.self)),
      numLayers * querySize * 4,
    );

    this.queryWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getQueryWeight(this.self)),
      numLayers * querySize * querySize * 4,
    );

    const keyValueSize = numKeyValueHeads * (querySize / numQueryHeads);

    this.keyWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getKeyWeight(this.self)),
      numLayers * keyValueSize * querySize * 4,
    );

    this.valueWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getValueWeight(this.self)),
      numLayers * keyValueSize * querySize * 4,
    );

    this.outputWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getOutputWeight(this.self)),
      numLayers * querySize * querySize * 4,
    );

    this.inputVector = new Float32Array(
      memory.buffer,
      checkNotNull(getInputVector(this.self)),
      querySize,
    );

    this.outputVector = new Float32Array(
      memory.buffer,
      checkNotNull(getOutputVector(this.self)),
      querySize,
    );
  }

  forward({ position, layer }: Readonly<{ position: number; layer: number }>): void {
    (this.wasmInstance.exports as WasmExports).forward(this.self, position, layer);
  }
}

interface WasmExports extends WebAssembly.Exports {
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
