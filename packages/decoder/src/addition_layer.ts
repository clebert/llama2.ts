interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  init(embedding_size: number): number;
  getInputVector(state: number): number;
  getOutputVector(state: number): number;
  forward(state: number): void;
}

export interface AdditionLayerInit {
  readonly embeddingSize: number;
}

export class AdditionLayer {
  static wasmModule: WebAssembly.Module | undefined;

  static async instantiate(init: AdditionLayerInit): Promise<AdditionLayer> {
    return new AdditionLayer(init, await WebAssembly.instantiate(AdditionLayer.wasmModule!));
  }

  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  readonly #wasmInstance: WebAssembly.Instance;
  readonly #wasmState: number;

  private constructor({ embeddingSize }: AdditionLayerInit, wasmInstance: WebAssembly.Instance) {
    const wasmExports = wasmInstance.exports as WasmExports;
    const wasmState = wasmExports.init(embeddingSize);

    this.inputVector = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getInputVector(wasmState),
      embeddingSize,
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
