interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  init(embedding_size: number, vocab_size: number): number;
  getInputVector(state: number): number;
  getNormWeightVector(state: number): number;
  getOutputWeightMatrix(state: number): number;
  getOutputVector(state: number): number;
  forward(state: number): number;
}

export interface LinearLayerInit {
  readonly embeddingSize: number;
  readonly vocabSize: number;
}

export class LinearLayer {
  static wasmModule: WebAssembly.Module | undefined;

  readonly inputVector: Float32Array;
  readonly normWeightVector: Float32Array;
  readonly outputWeightMatrix: Float32Array;
  readonly outputVector: Float32Array;

  readonly #wasmInstance: WebAssembly.Instance;
  readonly #wasmState: number;

  constructor({ embeddingSize, vocabSize }: LinearLayerInit) {
    const wasmInstance = new WebAssembly.Instance(LinearLayer.wasmModule!);
    const wasmExports = wasmInstance.exports as WasmExports;
    const wasmState = wasmExports.init(embeddingSize, vocabSize);

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

    this.outputWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getOutputWeightMatrix(wasmState),
      vocabSize * embeddingSize,
    );

    this.outputVector = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getOutputVector(wasmState),
      vocabSize,
    );

    this.#wasmInstance = wasmInstance;
    this.#wasmState = wasmState;
  }

  forward(): number {
    const wasmExports = this.#wasmInstance.exports as WasmExports;

    return wasmExports.forward(this.#wasmState);
  }
}
