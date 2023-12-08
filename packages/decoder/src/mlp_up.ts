import { checkNotNull } from './check_not_null.js';

export interface MlpUpInit {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly numLayers: number;
}

export class MlpUp {
  static wasmModule: WebAssembly.Module;

  static async create(init: MlpUpInit): Promise<MlpUp> {
    return new MlpUp(init, await WebAssembly.instantiate(this.wasmModule));
  }

  readonly normWeight: Uint8Array;
  readonly gateWeight: Uint8Array;
  readonly upWeight: Uint8Array;
  readonly inputVector: Float32Array;
  readonly outputVector: Float32Array;

  private readonly self: number;

  constructor(
    { inputSize, outputSize, numLayers }: MlpUpInit,
    private readonly wasmInstance: WebAssembly.Instance,
  ) {
    const {
      memory,
      create,
      getNormWeight,
      getGateWeight,
      getUpWeight,
      getInputVector,
      getOutputVector,
    } = wasmInstance.exports as WasmExports;

    this.self = create(inputSize, outputSize, numLayers);

    this.normWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getNormWeight(this.self)),
      numLayers * inputSize * 4,
    );

    this.gateWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getGateWeight(this.self)),
      numLayers * outputSize * inputSize * 4,
    );

    this.upWeight = new Uint8Array(
      memory.buffer,
      checkNotNull(getUpWeight(this.self)),
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
  }

  forward({ layer }: Readonly<{ layer: number }>): void {
    (this.wasmInstance.exports as WasmExports).forward(this.self, layer);
  }
}

interface WasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;

  create(inputSize: number, outputSize: number, numLayers: number): number;
  getNormWeight(self: number): number;
  getGateWeight(self: number): number;
  getUpWeight(self: number): number;
  getInputVector(self: number): number;
  getOutputVector(self: number): number;
  forward(self: number, layer: number): void;
}
