import { computeArgmax } from './compute_argmax.js';
import { Decoder } from '@llama2/decoder';
import { createDataSource, loadCheckpoint, loadHyperparams, loadVocab } from '@llama2/loader';
import { Tokenizer } from '@llama2/tokenizer';
import { open } from 'node:fs/promises';
import { stdout } from 'process';

export interface CompleteArgs {
  readonly modelPath: string;
  readonly prompt: string | undefined;
  readonly maxSequenceLength: number | undefined;
}

export async function complete(args: CompleteArgs): Promise<void> {
  const { modelPath, prompt = `` } = args;
  const file = await open(modelPath);

  const dataSource = createDataSource(
    file.readableWebStream().getReader() as ReadableStreamDefaultReader,
  );

  await dataSource.next();

  const hyperparams = await loadHyperparams(dataSource);
  const vocab = await loadVocab(dataSource, hyperparams.vocabSize);
  const maxSequenceLength = args.maxSequenceLength || hyperparams.maxSequenceLength;
  const checkpoint = await loadCheckpoint(dataSource, hyperparams, { maxSequenceLength });

  await file.close();

  const tokenizer = new Tokenizer(vocab);
  const decoder = await Decoder.instantiate(hyperparams, checkpoint);
  const promptTokenIds = tokenizer.encode(prompt, { bos: true });

  let nextTokenId = promptTokenIds.shift() ?? tokenizer.bosTokenId;

  for (let position = 0; position < maxSequenceLength; position += 1) {
    const tokenId = nextTokenId;
    const logits = decoder.decode(tokenId, position);

    nextTokenId = promptTokenIds.shift() ?? computeArgmax(logits);

    const nextToken = tokenizer.decode(nextTokenId, tokenId);

    if (nextToken === undefined) {
      break;
    }

    stdout.write(nextToken);
  }

  stdout.write(`\n`);
}
