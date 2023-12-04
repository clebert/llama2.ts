import type { Checkpoint, ModelConfig } from './create_checkpoint.js';

export interface ForwardOptions {
  readonly computeSoftmax?: boolean;
}

export class Decoder {
  constructor(
    readonly modelConfig: ModelConfig,
    readonly checkpoint: Checkpoint,
  ) {}

  forward(tokenId: number, position: number, options?: ForwardOptions): Float32Array {
    const { embeddingWeight, attention, mlpUp, mlpDown, linear } = this.checkpoint;
    const hiddenVector = mlpDown.outputVector;

    hiddenVector.set(
      new Float32Array(
        embeddingWeight.buffer,
        embeddingWeight.byteOffset + tokenId * hiddenVector.byteLength,
        hiddenVector.length,
      ),
    );

    for (let layer = 0; layer < this.modelConfig.numLayers; layer += 1) {
      attention.inputVector.set(hiddenVector);
      attention.forward(position, layer);

      hiddenVector.set(attention.outputVector);

      mlpUp.inputVector.set(hiddenVector);
      mlpUp.forward(layer);

      mlpDown.inputVector.set(mlpUp.outputVector);
      mlpDown.residualVector.set(hiddenVector);
      mlpDown.forward(layer);
    }

    linear.inputVector.set(hiddenVector);
    linear.forward(options?.computeSoftmax ?? false);

    return linear.outputVector;
  }
}
