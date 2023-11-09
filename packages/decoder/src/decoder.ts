import type { AttentionLayer } from './attention_layer.js';
import type { FnnLayer } from './fnn_layer.js';
import type { LinearLayer } from './linear_layer.js';

import { AdditionLayer } from './addition_layer.js';

export interface Config {
  readonly embeddingSize: number;
  readonly hiddenSize: number;
  readonly layerCount: number;
  readonly queryHeadCount: number;
  readonly keyValueHeadCount: number;
  readonly vocabSize: number;
  readonly maxSequenceLength: number;
  readonly sharedOutputWeight: boolean;
}

export interface Checkpoint {
  readonly embeddingVectors: readonly Float32Array[];
  readonly attentionLayers: readonly AttentionLayer[];
  readonly fnnLayers: readonly FnnLayer[];
  readonly linearLayer: LinearLayer;
}

export class Decoder {
  readonly #config: Config;
  readonly #checkpoint: Checkpoint;
  readonly #additionLayer: AdditionLayer;

  constructor(config: Config, checkpoint: Checkpoint) {
    this.#config = config;
    this.#checkpoint = checkpoint;
    this.#additionLayer = new AdditionLayer(config.embeddingSize);
  }

  decode(tokenId: number, position: number): Float32Array {
    const hiddenVector = this.#additionLayer.outputVector;

    hiddenVector.set(this.#checkpoint.embeddingVectors[tokenId]!);

    for (let index = 0; index < this.#config.layerCount; index += 1) {
      const attentionLayer = this.#checkpoint.attentionLayers[index]!;

      attentionLayer.inputVector.set(hiddenVector);
      attentionLayer.forward(position);

      this.#additionLayer.inputVector.set(attentionLayer.outputVector);
      this.#additionLayer.forward();

      const fnnLayer = this.#checkpoint.fnnLayers[index]!;

      fnnLayer.inputVector.set(hiddenVector);
      fnnLayer.forward();

      this.#additionLayer.inputVector.set(fnnLayer.outputVector);
      this.#additionLayer.forward();
    }

    this.#checkpoint.linearLayer.inputVector.set(hiddenVector);
    this.#checkpoint.linearLayer.forward();

    return this.#checkpoint.linearLayer.outputVector;
  }
}
