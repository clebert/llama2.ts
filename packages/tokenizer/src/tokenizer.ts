import type { Vocab, VocabEntry } from '@llama2/loader';

export interface TokenizerEncodeOptions {
  readonly bos?: boolean;
  readonly eos?: boolean;
}

export class Tokenizer {
  readonly unkTokenId = 0;
  readonly bosTokenId = 1;
  readonly eosTokenId = 2;

  readonly #vocab: Vocab;

  constructor(vocab: Vocab) {
    this.#vocab = vocab;

    this.#assert(`<unk>`, this.unkTokenId);
    this.#assert(`<s>`, this.bosTokenId);
    this.#assert(`</s>`, this.eosTokenId);
    this.#assert(`<0x00>`, 3);
    this.#assert(`<0x01>`, 4);
  }

  encode(input: string, options?: TokenizerEncodeOptions): number[] {
    if (input.trim() === ``) {
      return [];
    }

    const tokenIds: number[] = [];

    if (options?.bos) {
      tokenIds.push(this.bosTokenId);
    }

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

    if (options?.eos) {
      tokenIds.push(this.eosTokenId);
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

    return decodeHex(prevTokenId === this.bosTokenId && token[0] === ` ` ? token.slice(1) : token);
  }

  #assert(expectedToken: string, tokenId: number): void {
    const actualToken = this.#vocab.entriesByTokenId[tokenId]?.token;

    if (expectedToken !== actualToken) {
      throw new Error(
        `Unsupported vocab detected. Expected '${expectedToken}' at position ${tokenId} but found '${actualToken}' instead.`,
      );
    }
  }
}

function decodeHex(token: string): string {
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
