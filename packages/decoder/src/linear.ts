import { checkNotNull } from './check_not_null.js';

export interface LinearInit {
  readonly inputSize: number;
  readonly outputSize: number;
}

export class Linear {
  static wasmModule: WebAssembly.Module;

  static async create(init: LinearInit): Promise<Linear> {
    return new Linear(init, await WebAssembly.instantiate(this.wasmModule));
  }

  readonly normWeight: Uint8Array;
  readonly outputWeight: Uint8Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  private readonly self: number;

  constructor(
    { inputSize, outputSize }: LinearInit,
    private readonly wasmInstance: WebAssembly.Instance,
  ) {
    const { memory, create, getNormWeight, getOutputWeight, getInputVector, getOutputVector } =
      wasmInstance.exports as WasmExports;

    this.self = create(inputSize, outputSize);

    this.normWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getNormWeight(this.self)),
      inputSize * 4,
    );

    this.outputWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getOutputWeight(this.self)),
      outputSize * inputSize * 4,
    );

    this.inputVector = new Float32Array(
      memory.buffer,
      checkNotNull(getInputVector(this.self)),
      inputSize,
    );

    this.outputVector = new Float32Array(
      memory.buffer,
      checkNotNull(getOutputVector(this.self)),
      outputSize,
    );
  }

  forward(): void {
    (this.wasmInstance.exports as WasmExports).forward(this.self);
  }

  computeSoftmax(): void {
    (this.wasmInstance.exports as WasmExports).computeSoftmax(this.self);
  }
}

interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  create(inputSize: number, outputSize: number): number;
  getNormWeight(self: number): number;
  getOutputWeight(self: number): number;
  getInputVector(self: number): number;
  getOutputVector(self: number): number;
  forward(self: number): void;
  computeSoftmax(self: number): void;
}
