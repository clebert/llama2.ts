import { computeArgmax } from './compute_argmax.js';
import { loadWasmModule } from './load_wasm_module.js';
import { Attention, Decoder, Linear, MlpDown, MlpUp } from '@llama2/decoder';
import { createDataSource, loadCheckpoint, loadModelConfig, loadVocab } from '@llama2/loader';
import { Tokenizer } from '@llama2/tokenizer';

async function main(): Promise<void> {
  Attention.wasmSingleton = await WebAssembly.instantiate(await loadWasmModule(`attention`));
  Linear.wasmSingleton = await WebAssembly.instantiate(await loadWasmModule(`linear`));
  MlpDown.wasmSingleton = await WebAssembly.instantiate(await loadWasmModule(`mlp_down`));
  MlpUp.wasmSingleton = await WebAssembly.instantiate(await loadWasmModule(`mlp_up`));

  const response = await fetch(`/static/models/tinystories_15m_v1.bin`);
  const reader = response.body!.getReader();
  const dataSource = createDataSource(reader);

  await dataSource.next();

  const modelConfig = await loadModelConfig(dataSource);

  console.log({ modelConfig });

  const vocab = await loadVocab(dataSource, modelConfig);
  const checkpoint = await loadCheckpoint(dataSource, modelConfig);

  const tokenizer = new Tokenizer(vocab);
  const decoder = new Decoder({ numLayers: modelConfig.numLayers, checkpoint });

  const run = () => {
    const tokenIds = [tokenizer.bosTokenId];

    let totalTime = 0;

    while (tokenIds.length <= modelConfig.maxSequenceLength) {
      const position = tokenIds.length - 1;
      const tokenId = tokenIds[position]!;

      let startTime = 0;

      if (position > 0) {
        startTime = performance.now();
      }

      const logits = decoder.decode({ tokenId, position });

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
  };

  run();
}

main().catch((error) => {
  console.error(error);
});
