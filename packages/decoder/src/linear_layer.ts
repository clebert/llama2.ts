interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  init(input_vector_len: number, output_vector_len: number): number;
  getOutputWeightMatrix(state: number): number;
  getNormWeightVector(state: number): number;
  getInputVector(state: number): number;
  getOutputVector(state: number): number;
  forward(state: number): number;
}

export interface LinearLayerInit {
  readonly inputVectorLength: number;
  readonly outputVectorLength: number;
}

export class LinearLayer {
  static wasmModule: WebAssembly.Module | undefined;

  readonly outputWeightMatrix: Float32Array;
  readonly normWeightVector: Float32Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  readonly #wasmInstance: WebAssembly.Instance;
  readonly #wasmState: number;

  constructor({ inputVectorLength, outputVectorLength }: LinearLayerInit) {
    const wasmInstance = new WebAssembly.Instance(LinearLayer.wasmModule!);
    const wasmExports = wasmInstance.exports as WasmExports;
    const wasmState = wasmExports.init(inputVectorLength, outputVectorLength);

    this.outputWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getOutputWeightMatrix(wasmState),
      outputVectorLength * inputVectorLength,
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
      outputVectorLength,
    );

    this.#wasmInstance = wasmInstance;
    this.#wasmState = wasmState;
  }

  forward(): number {
    const wasmExports = this.#wasmInstance.exports as WasmExports;

    return wasmExports.forward(this.#wasmState);
  }
}
