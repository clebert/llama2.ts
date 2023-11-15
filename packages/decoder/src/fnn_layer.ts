interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  init(embedding_size: number, hidden_size: number): number;
  getInputVector(state: number): number;
  getNormWeightVector(state: number): number;
  getGateWeightMatrix(state: number): number;
  getUpWeightMatrix(state: number): number;
  getDownWeightMatrix(state: number): number;
  getOutputVector(state: number): number;
  forward(state: number): void;
}

export interface FnnLayerInit {
  readonly embeddingSize: number;
  readonly hiddenSize: number;
}

export class FnnLayer {
  static wasmModule: WebAssembly.Module | undefined;

  static async instantiate(init: FnnLayerInit): Promise<FnnLayer> {
    return new FnnLayer(init, await WebAssembly.instantiate(FnnLayer.wasmModule!));
  }

  readonly inputVector: Float32Array;
  readonly normWeightVector: Float32Array;
  readonly gateWeightMatrix: Float32Array;
  readonly upWeightMatrix: Float32Array;
  readonly downWeightMatrix: Float32Array;
  readonly outputVector: Float32Array;

  readonly #wasmInstance: WebAssembly.Instance;
  readonly #wasmState: number;

  private constructor(
    { embeddingSize, hiddenSize }: FnnLayerInit,
    wasmInstance: WebAssembly.Instance,
  ) {
    const wasmExports = wasmInstance.exports as WasmExports;
    const wasmState = wasmExports.init(embeddingSize, hiddenSize);

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

    this.gateWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getGateWeightMatrix(wasmState),
      hiddenSize * embeddingSize,
    );

    this.upWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getUpWeightMatrix(wasmState),
      hiddenSize * embeddingSize,
    );

    this.downWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getDownWeightMatrix(wasmState),
      embeddingSize * hiddenSize,
    );

    this.outputVector = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getOutputVector(wasmState),
      embeddingSize,
    );

    this.#wasmInstance = wasmInstance;
    this.#wasmState = wasmState;
  }

  forward(): void {
    const wasmExports = this.#wasmInstance.exports as WasmExports;

    wasmExports.forward(this.#wasmState);
  }
}
