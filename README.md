# Llama2.ts

> Inference Llama 2 in pure TypeScript and Zig.

**NOTE:** This project is currently a Work in Progress (WIP). Please be aware that everything can
change at any time.

## Getting Started

Install dependencies:

```sh
npm i
```

### Node.js

Compile and run the program:

```
npm run compile
```

```
npm run start:node -- complete models/tinystories_15m_v1.bin
```

Output:

```
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine.
```

### Browser

Build the website and start the server:

```
npm run build
```

```
npm run start:browser
```

Open: http://localhost:3000

## Using Llama 2 Models from Hugging Face

Install `git-lfs` and clone the
[TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T)
model from Hugging Face:

```sh
git lfs install
```

```sh
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T
```

Install the necessary Python packages and convert the Hugging Face model:

```sh
pip3 install -r requirements.txt
```

```sh
python3 convert_hf_model.py /path/to/TinyLlama-1.1B models/tiny_llama_v1.bin
```

Compile and run the program:

```sh
npm run compile
```

```sh
npm run start:node -- complete models/tiny_llama_v1.bin --prompt "The number 42 is"
```

## Model Binary Data Format

```
+--------------------------+
| Model Config             |
+--------------------------+
| Vocab Entry 0            |
+--------------------------+
| Vocab Entry ..           |
+--------------------------+
| Checkpoint               |
+--------------------------+
```

### Model Config (`256` bytes)

| Element               | Size                         | Example |
| --------------------- | ---------------------------- | ------- |
| `version`             | `1` × `u8`                   | 2       |
| `modelTypeByteLength` | `1` × `i32`                  | 5       |
| `modelType`           | `modelTypeByteLength` × `u8` | llama   |
| `hiddenSize`          | `1` × `i32`                  | 4096    |
| `intermediateSize`    | `1` × `i32`                  | 11008   |
| `maxSequenceLength`   | `1` × `i32`                  | 4096    |
| `vocabSize`           | `1` × `i32`                  | 32000   |
| `numLayers`           | `1` × `i32`                  | 32      |
| `numQueryHeads`       | `1` × `i32`                  | 32      |
| `numKeyValueHeads`    | `1` × `i32`                  | 32      |
| `sharedOutputWeight`  | `1` × `u8`                   | 0       |

**Note:** `keyValueSize` = `numKeyValueHeads` × (`hiddenSize` ÷ `numQueryHeads`)

### Vocab Entry (`0` .. `vocabSize`)

| Element           | Size                     |
| ----------------- | ------------------------ |
| `score`           | `1` × `f32`              |
| `tokenByteLength` | `1` × `i32`              |
| `token`           | `tokenByteLength` × `u8` |

### Checkpoint (`version` = `1`)

| Element                 | Size                                                    | Condition                  |
| ----------------------- | ------------------------------------------------------- | -------------------------- |
| `embeddingWeight`       | `vocabSize` × `hiddenSize` × `f32`                      |                            |
| `attentionNormWeight`   | `numLayers` × `hiddenSize` × `f32`                      |                            |
| `attentionQueryWeight`  | `numLayers` × `hiddenSize` × `hiddenSize` × `f32`       |                            |
| `attentionKeyWeight`    | `numLayers` × `keyValueSize` × `hiddenSize` × `f32`     |                            |
| `attentionValueWeight`  | `numLayers` × `keyValueSize` × `hiddenSize` × `f32`     |                            |
| `attentionOutputWeight` | `numLayers` × `hiddenSize` × `hiddenSize` × `f32`       |                            |
| `mlpNormWeight`         | `numLayers` × `hiddenSize` × `f32`                      |                            |
| `mlpGateWeight`         | `numLayers` × `intermediateSize` × `hiddenSize` × `f32` |                            |
| `mlpUpWeight`           | `numLayers` × `intermediateSize` × `hiddenSize` × `f32` |                            |
| `mlpDownWeight`         | `numLayers` × `hiddenSize` × `intermediateSize` × `f32` |                            |
| `linearNormWeight`      | `hiddenSize` × `f32`                                    |                            |
| `linearOutputWeight`    | `vocabSize` × `hiddenSize` × `f32`                      | `sharedOutputWeight` = `0` |
