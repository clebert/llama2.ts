import type { Attention } from './attention.js';
import type { Linear } from './linear.js';
import type { MlpDown } from './mlp_down.js';
import type { MlpUp } from './mlp_up.js';

export interface DecoderInit {
  readonly numLayers: number;
  readonly checkpoint: Checkpoint;
}

export interface Checkpoint {
  readonly embeddingWeight: Uint8Array;
  readonly attention: Attention;
  readonly mlpUp: MlpUp;
  readonly mlpDown: MlpDown;
  readonly linear: Linear;
}

export class Decoder {
  constructor(private readonly init: DecoderInit) {}

  decode({
    tokenId,
    position,
    softmax,
  }: Readonly<{ tokenId: number; position: number; softmax?: boolean }>): Float32Array {
    const {
      numLayers,
      checkpoint: { embeddingWeight, attention, mlpUp, mlpDown, linear },
    } = this.init;

    const hiddenVector = mlpDown.outputVector;

    hiddenVector.set(
      new Float32Array(
        embeddingWeight.buffer,
        embeddingWeight.byteOffset + tokenId * hiddenVector.byteLength,
        hiddenVector.length,
      ),
    );

    for (let layer = 0; layer < numLayers; layer += 1) {
      attention.inputVector.set(hiddenVector);
      attention.forward({ position, layer });

      hiddenVector.set(attention.outputVector);

      mlpUp.inputVector.set(hiddenVector);
      mlpUp.forward({ layer });

      mlpDown.inputVector.set(mlpUp.outputVector);
      mlpDown.residualVector.set(hiddenVector);
      mlpDown.forward({ layer });
    }

    linear.inputVector.set(hiddenVector);
    linear.forward();

    if (softmax) {
      linear.computeSoftmax();
    }

    return linear.outputVector;
  }
}
