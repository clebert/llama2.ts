import { computeArgmax } from './compute_argmax.js';
import { loadWasmModule } from './load_wasm_module.js';
import { Decoder, createCheckpoint } from '@llama2/decoder';
import { createDataSource, loadCheckpoint, loadHeader, loadVocab } from '@llama2/loader';
import { Tokenizer } from '@llama2/tokenizer';
import { open } from 'node:fs/promises';
import { stdout } from 'process';

export interface CompleteArgs {
  readonly modelPath: string;
  readonly prompt: string | undefined;
  readonly maxSequenceLength: number | undefined;
}

export async function complete({
  modelPath,
  prompt = ``,
  maxSequenceLength,
}: CompleteArgs): Promise<void> {
  const file = await open(modelPath);

  const dataSource = createDataSource(
    file.readableWebStream().getReader() as ReadableStreamDefaultReader,
  );

  await dataSource.next();

  const header = await loadHeader(dataSource);
  const vocab = await loadVocab(dataSource, header);
  const { modelConfig } = header;

  const checkpoint = await createCheckpoint(modelConfig, {
    attention: await loadWasmModule(`attention`),
    mlpUp: await loadWasmModule(`mlp_up`),
    mlpDown: await loadWasmModule(`mlp_down`),
    linear: await loadWasmModule(`linear`),
  });

  await loadCheckpoint(dataSource, header, checkpoint);
  await file.close();

  const decoder = new Decoder(modelConfig, checkpoint);
  const tokenizer = new Tokenizer(vocab);
  const promptTokenIds = tokenizer.encode(prompt, { bos: true });

  let nextTokenId = promptTokenIds.shift() ?? tokenizer.bosTokenId;
  let totalTime = 0;
  let numTokens = 0;

  for (
    let position = 0;
    position < (maxSequenceLength ?? modelConfig.maxSequenceLength);
    position += 1
  ) {
    const tokenId = nextTokenId;
    const startTime = performance.now();
    const logits = decoder.forward(tokenId, position);

    totalTime += performance.now() - startTime;
    nextTokenId = promptTokenIds.shift() ?? computeArgmax(logits);

    const nextToken = tokenizer.decode(nextTokenId, tokenId);

    numTokens += 1;

    if (nextToken === undefined) {
      break;
    }

    stdout.write(nextToken);
  }

  stdout.write(`\n`);

  const averageTime = totalTime / numTokens;

  console.log(`\n\nachieved: ${(1000 / averageTime).toFixed(3)} tok/s`);
}
