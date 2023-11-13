# Llama2.ts

> Inference Llama 2 in pure TypeScript and Zig.

**NOTE:** This project is currently a Work in Progress (WIP). Please be aware that everything can
change at any time. As it stands now, it may not be highly useful for other users. Stay tuned for
more updates as the project progresses.

## Getting Started

**Install dependencies:**

```sh
npm i
```

### Node.js

**Compile sources:**

```
npm run compile
```

**Start inference:**

```
npm run start:node
```

```
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine.

achieved: 556.172 tok/s
```

### Browser

**NOTE:** I have only been able to run it successfully in Chrome so far. Safari crashes.

**Build bundles:**

```
npm run build
```

**Start server:**

```
npm run start:browser
```

**Open:** http://localhost:3000

## Model Files

The transformer model files located in the `models` directory were
[trained](https://github.com/karpathy/llama2.c#models) on the TinyStories dataset by Andrej Karpathy
for his implementation of Llama 2 in C. These files have been adapted into a binary data format
specifically optimized for browser streaming. If interested, larger versions can be found
[here](https://huggingface.co/clebert/tinystories). For more information on the binary data format,
please refer to the documentation provided below.

## Model Binary Data Format

```
+--------------------------+
| HEADER (HYPERPARAMETERS) |
+--------------------------+
| VOCAB ENTRY 0            |
+--------------------------+
| VOCAB ENTRY ..           |
+--------------------------+
| EMBEDDING 0              |
+--------------------------+
| EMBEDDING ..             |
+--------------------------+
| ATTENTION LAYER 0        |
+--------------------------+
| ATTENTION LAYER ..       |
+--------------------------+
| FNN LAYER 0              |
+--------------------------+
| FNN LAYER ..             |
+--------------------------+
| LINEAR LAYER             |
+--------------------------+
```

**NOTE:** All `i32` and `f32` data types are in little-endian format and matrices are organized in a
row-first order.

### Header (`256` bytes)

| Element           | Type       | Example  |
| ----------------- | ---------- | -------- |
| dataFormatMagic   | `6` x `u8` | "llama2" |
| dataFormatVersion | `1` x `u8` | 1        |

#### Hyperparameters

| Element            | Type        | Example |
| ------------------ | ----------- | ------- |
| embeddingSize      | `1` x `i32` | 4096    |
| hiddenSize         | `1` x `i32` | 11008   |
| keyValueSize       | `1` x `i32` | 4096    |
| layerCount         | `1` x `i32` | 32      |
| queryHeadCount     | `1` x `i32` | 32      |
| vocabSize          | `1` x `i32` | 32000   |
| maxSequenceLength  | `1` x `i32` | 4096    |
| sharedOutputWeight | `1` x `u8`  | 0       |

### Vocab Entry (`0` .. `vocabSize`)

| Element         | Type                     | Example |
| --------------- | ------------------------ | ------- |
| score           | `1` x `f32`              | -10735  |
| tokenByteLength | `1` x `i32`              | 5       |
| token           | `tokenByteLength` x `u8` | "Hello" |

### Embedding (`0` .. `vocabSize`)

| Element         | Type                    |
| --------------- | ----------------------- |
| embeddingVector | `embeddingSize` x `f32` |

### Attention Layer (`0` .. `layerCount`)

| Element            | Type                                      |
| ------------------ | ----------------------------------------- |
| normWeightVector   | `embeddingSize` x `f32`                   |
| queryWeightMatrix  | `embeddingSize` x `embeddingSize` x `f32` |
| keyWeightMatrix    | `keyValueSize` x `embeddingSize` x `f32`  |
| valueWeightMatrix  | `keyValueSize` x `embeddingSize` x `f32`  |
| outputWeightMatrix | `embeddingSize` x `embeddingSize` x `f32` |

### FNN Layer (`0` .. `layerCount`)

| Element          | Type                                   |
| ---------------- | -------------------------------------- |
| normWeightVector | `embeddingSize` x `f32`                |
| gateWeightMatrix | `hiddenSize` x `embeddingSize` x `f32` |
| upWeightMatrix   | `hiddenSize` x `embeddingSize` x `f32` |
| downWeightMatrix | `embeddingSize` x `hiddenSize` x `f32` |

### Linear Layer

| Element            | Type                                  |
| ------------------ | ------------------------------------- |
| normWeightVector   | `embeddingSize` x `f32`               |
| outputWeightMatrix | `vocabSize` x `embeddingSize` x `f32` |
