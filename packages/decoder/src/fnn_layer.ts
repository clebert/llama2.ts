interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  init(input_vector_len: number, hidden_vector_len: number): number;
  getGateWeightMatrix(state: number): number;
  getUpWeightMatrix(state: number): number;
  getDownWeightMatrix(state: number): number;
  getNormWeightVector(state: number): number;
  getInputVector(state: number): number;
  getOutputVector(state: number): number;
  forward(state: number): number;
}

export interface FnnLayerInit {
  readonly inputVectorLength: number;
  readonly hiddenVectorLength: number;
}

export class FnnLayer {
  static wasmModule: WebAssembly.Module | undefined;

  readonly gateWeightMatrix: Float32Array;
  readonly upWeightMatrix: Float32Array;
  readonly downWeightMatrix: Float32Array;
  readonly normWeightVector: Float32Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  readonly #wasmInstance: WebAssembly.Instance;
  readonly #wasmState: number;

  constructor({ inputVectorLength, hiddenVectorLength }: FnnLayerInit) {
    const wasmInstance = new WebAssembly.Instance(FnnLayer.wasmModule!);
    const wasmExports = wasmInstance.exports as WasmExports;
    const wasmState = wasmExports.init(inputVectorLength, hiddenVectorLength);

    this.gateWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getGateWeightMatrix(wasmState),
      hiddenVectorLength * inputVectorLength,
    );

    this.upWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getUpWeightMatrix(wasmState),
      hiddenVectorLength * inputVectorLength,
    );

    this.downWeightMatrix = new Float32Array(
      wasmExports.memory.buffer,
      wasmExports.getDownWeightMatrix(wasmState),
      inputVectorLength * hiddenVectorLength,
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

  forward(): number {
    const wasmExports = this.#wasmInstance.exports as WasmExports;

    return wasmExports.forward(this.#wasmState);
  }
}
