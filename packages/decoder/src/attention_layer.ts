interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  init(
    input_vector_len: number,
    query_head_count: number,
    key_value_head_count: number,
    sequence_len: number,
  ): number;

  getQueryWeightMatrix(state: number): number;
  getKeyWeightMatrix(state: number): number;
  getValueWeightMatrix(state: number): number;
  getOutputWeightMatrix(state: number): number;
  getNormWeightVector(state: number): number;
  getInputVector(state: number): number;
  getOutputVector(state: number): number;
  forward(state: number, position: number): void;
}

export interface AttentionLayerInit {
  readonly inputVectorLength: number;
  readonly queryHeadCount: number;
  readonly keyValueHeadCount: number;
  readonly sequenceLength: number;
}

export class AttentionLayer {
  static wasmModule: WebAssembly.Module | undefined;

  readonly queryWeightMatrix: Float32Array;
  readonly keyWeightMatrix: Float32Array;
  readonly valueWeightMatrix: Float32Array;
  readonly outputWeightMatrix: Float32Array;
  readonly normWeightVector: Float32Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  readonly #wasmInstance: WebAssembly.Instance;
  readonly #wasmState: number;

  constructor({
    inputVectorLength,
    queryHeadCount,
    keyValueHeadCount,
    sequenceLength,
  }: AttentionLayerInit) {
    const wasmInstance = new WebAssembly.Instance(AttentionLayer.wasmModule!);
    const wasmExports = wasmInstance.exports as WasmExports;

    const wasmState = wasmExports.init(
      inputVectorLength,
      queryHeadCount,
      keyValueHeadCount,
      sequenceLength,
    );

    this.queryWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getQueryWeightMatrix(wasmState),
      inputVectorLength * inputVectorLength,
    );

    const headSize = inputVectorLength / queryHeadCount;
    const keyValueSize = keyValueHeadCount * headSize;

    this.keyWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getKeyWeightMatrix(wasmState),
      keyValueSize * inputVectorLength,
    );

    this.valueWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getValueWeightMatrix(wasmState),
      keyValueSize * inputVectorLength,
    );

    this.outputWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getOutputWeightMatrix(wasmState),
      inputVectorLength * inputVectorLength,
    );

    this.normWeightVector = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getNormWeightVector(wasmState),
      inputVectorLength,
    );

    this.inputVector = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getInputVector(wasmState),
      inputVectorLength,
    );

    this.outputVector = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getOutputVector(wasmState),
      inputVectorLength,
    );

    this.#wasmInstance = wasmInstance;
    this.#wasmState = wasmState;
  }

  forward(position: number): void {
    const wasmExports = this.#wasmInstance.exports as WasmExports;

    wasmExports.forward(this.#wasmState, position);
  }
}
