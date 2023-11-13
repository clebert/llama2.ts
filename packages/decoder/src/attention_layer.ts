interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  init(
    embedding_size: number,
    key_value_size: number,
    query_head_count: number,
    sequence_len: number,
  ): number;

  getInputVector(state: number): number;
  getNormWeightVector(state: number): number;
  getQueryWeightMatrix(state: number): number;
  getKeyWeightMatrix(state: number): number;
  getValueWeightMatrix(state: number): number;
  getOutputWeightMatrix(state: number): number;
  getOutputVector(state: number): number;
  forward(state: number, position: number): void;
}

export interface AttentionLayerInit {
  readonly embeddingSize: number;
  readonly keyValueSize: number;
  readonly queryHeadCount: number;
  readonly sequenceLength: number;
}

export class AttentionLayer {
  static wasmModule: WebAssembly.Module | undefined;

  static async instantiate(init: AttentionLayerInit): Promise<AttentionLayer> {
    return new AttentionLayer(init, await WebAssembly.instantiate(AttentionLayer.wasmModule!));
  }

  readonly inputVector: Float32Array;
  readonly normWeightVector: Float32Array;
  readonly queryWeightMatrix: Float32Array;
  readonly keyWeightMatrix: Float32Array;
  readonly valueWeightMatrix: Float32Array;
  readonly outputWeightMatrix: Float32Array;
  readonly outputVector: Float32Array;

  readonly #wasmInstance: WebAssembly.Instance;
  readonly #wasmState: number;

  private constructor(
    { embeddingSize, keyValueSize, queryHeadCount, sequenceLength }: AttentionLayerInit,
    wasmInstance: WebAssembly.Instance,
  ) {
    const wasmExports = wasmInstance.exports as WasmExports;
    const wasmState = wasmExports.init(embeddingSize, keyValueSize, queryHeadCount, sequenceLength);

    this.inputVector = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getInputVector(wasmState),
      embeddingSize,
    );

    this.normWeightVector = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getNormWeightVector(wasmState),
      embeddingSize,
    );

    this.queryWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getQueryWeightMatrix(wasmState),
      embeddingSize * embeddingSize,
    );

    this.keyWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getKeyWeightMatrix(wasmState),
      keyValueSize * embeddingSize,
    );

    this.valueWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getValueWeightMatrix(wasmState),
      keyValueSize * embeddingSize,
    );

    this.outputWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getOutputWeightMatrix(wasmState),
      embeddingSize * embeddingSize,
    );

    this.outputVector = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getOutputVector(wasmState),
      embeddingSize,
    );

    this.#wasmInstance = wasmInstance;
    this.#wasmState = wasmState;
  }

  forward(position: number): void {
    const wasmExports = this.#wasmInstance.exports as WasmExports;

    wasmExports.forward(this.#wasmState, position);
  }
}
