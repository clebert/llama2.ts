import type { Vocab, VocabEntry } from '@llama2/loader';

export interface TokenizerDecodeOptions {
  readonly bos?: boolean;
}

export class Tokenizer {
  readonly unkTokenId: number;
  readonly bosTokenId: number;
  readonly eosTokenId: number;

  readonly #vocab: Vocab;

  constructor(vocab: Vocab) {
    this.unkTokenId = vocab.entriesByToken.get(`<unk>`)!.tokenId;
    this.bosTokenId = vocab.entriesByToken.get(`\n<s>\n`)!.tokenId; // TODO: use "<s>"
    this.eosTokenId = vocab.entriesByToken.get(`\n</s>\n`)!.tokenId; // TODO: use "</s>"

    this.#vocab = vocab;
  }

  encode(input: string): number[] {
    if (input.trim() === ``) {
      return [];
    }

    const tokenIds: number[] = [];

    // TODO: use "▁"
    for (const token of [...` ${input}`]) {
      const entry = this.#vocab.entriesByToken.get(token);

      if (entry) {
        tokenIds.push(entry.tokenId);
      } else {
        for (const byte of new TextEncoder().encode(token)) {
          tokenIds.push(byte + 3);
        }
      }
    }

    while (true) {
      let best: { readonly entry: VocabEntry; readonly index: number } | undefined;

      for (let index = 0; index < tokenIds.length - 1; index += 1) {
        const entry1 = this.#vocab.entriesByTokenId[tokenIds[index]!]!;
        const entry2 = this.#vocab.entriesByTokenId[tokenIds[index + 1]!]!;
        const entry3 = this.#vocab.entriesByToken.get(entry1.token + entry2.token);

        if (entry3 && (!best || entry3.score > best.entry.score)) {
          best = { entry: entry3, index };
        }
      }

      if (!best) {
        break;
      }

      tokenIds[best.index] = best.entry.tokenId;

      tokenIds.splice(best.index + 1, 1);
    }

    return tokenIds;
  }

  decode(tokenId: number, prevTokenId?: number): string | undefined {
    if (tokenId === this.unkTokenId || tokenId === this.bosTokenId || tokenId === this.eosTokenId) {
      return undefined;
    }

    const { token } = this.#vocab.entriesByTokenId[tokenId]!;

    return format(prevTokenId === this.bosTokenId && /^\s/.test(token) ? token.slice(1) : token);
  }
}

function format(token: string): string {
  const result = token.match(/^<0x([0-9A-F]+)>$/);

  if (!result) {
    return token;
  }

  const code = parseInt(result[1]!, 16);
  const char = String.fromCharCode(code);

  return isPrintable(code) || isWhitespace(char) ? char : token;
}

function isPrintable(code: number): boolean {
  return code >= 32 && code <= 126;
}

function isWhitespace(char: string): boolean {
  return /\s/.test(char);
}
