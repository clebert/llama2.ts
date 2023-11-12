import { computeArgmax } from './compute_argmax.js';
import { loadWasmModule } from './load_wasm_module.js';
import { AdditionLayer, AttentionLayer, Decoder, FnnLayer, LinearLayer } from '@llama2/decoder';
import { createDataSource, loadCheckpoint, loadHyperparams, loadVocab } from '@llama2/loader';
import { Tokenizer } from '@llama2/tokenizer';

async function main(): Promise<void> {
  AdditionLayer.wasmModule = await loadWasmModule(`addition_layer`);
  AttentionLayer.wasmModule = await loadWasmModule(`attention_layer`);
  FnnLayer.wasmModule = await loadWasmModule(`fnn_layer`);
  LinearLayer.wasmModule = await loadWasmModule(`linear_layer`);

  const response = await fetch(`/static/models/tinystories_15m.bin`);
  const reader = response.body!.getReader();
  const dataSource = createDataSource(reader);

  await dataSource.next();

  const hyperparams = await loadHyperparams(dataSource);

  console.log({ hyperparams });

  const vocab = await loadVocab(dataSource, hyperparams.vocabSize);
  const sequenceLength = hyperparams.maxSequenceLength;
  const checkpoint = await loadCheckpoint(dataSource, hyperparams, { sequenceLength });

  const tokenizer = new Tokenizer(vocab);
  const decoder = await Decoder.instantiate(hyperparams, checkpoint);

  const tokenIds = [tokenizer.bosTokenId];

  let totalTime = 0;

  while (tokenIds.length <= sequenceLength) {
    const position = tokenIds.length - 1;
    const tokenId = tokenIds[position]!;

    let startTime = 0;

    if (position > 0) {
      startTime = performance.now();
    }

    const logits = decoder.decode(tokenId, position);

    if (startTime > 0) {
      totalTime += performance.now() - startTime;
    }

    const nextTokenId = computeArgmax(logits);
    const nextToken = tokenizer.decode(nextTokenId, tokenId);

    if (nextToken === undefined) {
      break;
    }

    self.postMessage(nextToken);

    tokenIds.push(nextTokenId);
  }

  if (tokenIds.length > 1) {
    const averageTime = totalTime / (tokenIds.length - 1);

    console.log(`\n\nachieved: ${(1000 / averageTime).toFixed(3)} tok/s`);
  }
}

main().catch((error) => {
  console.error(error);
});
