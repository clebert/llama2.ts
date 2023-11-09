interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  init(input_vector_len: number): number;
  getInputVector(state: number): number;
  getOutputVector(state: number): number;
  forward(state: number): void;
}

export class AdditionLayer {
  static wasmModule: WebAssembly.Module | undefined;

  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  readonly #wasmInstance: WebAssembly.Instance;
  readonly #wasmState: number;

  constructor(inputVectorLength: number) {
    const wasmInstance = new WebAssembly.Instance(AdditionLayer.wasmModule!);
    const wasmExports = wasmInstance.exports as WasmExports;
    const wasmState = wasmExports.init(inputVectorLength);

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

  forward(): void {
    const wasmExports = this.#wasmInstance.exports as WasmExports;

    wasmExports.forward(this.#wasmState);
  }
}
