import { checkNotNull } from './check_not_null.js';

export interface MlpDownInit {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly numLayers: number;
}

export class MlpDown {
  static wasmModule: WebAssembly.Module;

  static async create(init: MlpDownInit): Promise<MlpDown> {
    return new MlpDown(init, await WebAssembly.instantiate(this.wasmModule));
  }

  readonly downWeight: Uint8Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;
  readonly residualVector: Float32Array;

  private readonly self: number;

  constructor(
    { inputSize, outputSize, numLayers }: MlpDownInit,
    private readonly wasmInstance: WebAssembly.Instance,
  ) {
    const { memory, create, getDownWeight, getInputVector, getOutputVector, getResidualVector } =
      wasmInstance.exports as WasmExports;

    this.self = create(inputSize, outputSize, numLayers);

    this.downWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getDownWeight(this.self)),
      numLayers * outputSize * inputSize * 4,
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

    this.residualVector = new Float32Array(
      memory.buffer,
      checkNotNull(getResidualVector(this.self)),
      outputSize,
    );
  }

  forward({ layer }: Readonly<{ layer: number }>): void {
    (this.wasmInstance.exports as WasmExports).forward(this.self, layer);
  }
}

interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  create(inputSize: number, outputSize: number, numLayers: number): number;
  getDownWeight(self: number): number;
  getInputVector(self: number): number;
  getOutputVector(self: number): number;
  getResidualVector(self: number): number;
  forward(self: number, layer: number): void;
}
