import argparse
import os
import struct
import torch
from transformers import AutoModelForCausalLM
from sentencepiece import SentencePieceProcessor


def serialize_f32(file, tensor):
    tensor_f32 = tensor.detach().cpu().view(-1).to(torch.float32).numpy()

    file.write(struct.pack(f"{len(tensor_f32)}f", *tensor_f32))


# https://github.com/huggingface/transformers/blob/5c081e29930466ecf9a478727039d980131076d9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122C28-L122C35
def unpermute(tensor, n_heads, dim1, dim2):
    return (
        tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def write_model_file():
    model = AutoModelForCausalLM.from_pretrained(args.input_model_path)

    if model.config.model_type != "llama":
        parser.error("Expected llama model")

    if model.config.rope_theta != 10000:
        parser.error("Expected a RoPE frequency base of 10000")

    state = model.state_dict()

    embedding_vectors = state["model.embed_tokens.weight"]
    linear_norm_weight_vector = state["model.norm.weight"]
    linear_output_weight_matrix = state[f"lm_head.weight"]

    embedding_size = model.config.hidden_size
    hidden_size = model.config.intermediate_size
    layer_count = model.config.num_hidden_layers
    query_head_count = model.config.num_attention_heads
    key_value_head_count = model.config.num_key_value_heads
    vocab_size = model.config.vocab_size
    max_sequence_length = model.config.max_position_embeddings
    shared_output_weight = torch.equal(embedding_vectors, linear_output_weight_matrix)

    head_size = embedding_size // query_head_count
    key_value_size = key_value_head_count * head_size

    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)

    output_file = open(args.output_model_path, "wb")

    # Header #######################################################################################

    output_file.write("llama2".encode("utf-8"))
    output_file.write(struct.pack("B", 1))

    # Hyperparameters

    output_file.write(
        struct.pack(
            "iiiiiii",
            embedding_size,
            hidden_size,
            key_value_size,
            layer_count,
            query_head_count,
            vocab_size,
            max_sequence_length,
        )
    )

    output_file.write(struct.pack("B", int(shared_output_weight)))

    output_file.write(b"\0" * (256 - output_file.tell()))

    # Vocab entries ################################################################################

    spp = SentencePieceProcessor(
        model_file=os.path.join(args.input_model_path, "tokenizer.model")
    )

    scores, tokens = [], []

    for token_id in range(spp.vocab_size()):
        scores.append(spp.get_score(token_id))
        tokens.append(spp.id_to_piece(token_id).encode("utf-8"))

    for score, token in zip(scores, tokens):
        output_file.write(struct.pack("fi", score, len(token)))
        output_file.write(token)

    # Embeddings ###################################################################################

    serialize_f32(output_file, embedding_vectors)

    # Attention layers #############################################################################

    for layer in range(layer_count):
        attention_norm_weight_vector = state[
            f"model.layers.{layer}.input_layernorm.weight"
        ]

        serialize_f32(output_file, attention_norm_weight_vector)

        attention_query_weight_matrix = state[
            f"model.layers.{layer}.self_attn.q_proj.weight"
        ]

        serialize_f32(
            output_file,
            unpermute(
                attention_query_weight_matrix,
                query_head_count,
                embedding_size,
                embedding_size,
            ),
        )

        attention_key_weight_matrix = state[
            f"model.layers.{layer}.self_attn.k_proj.weight"
        ]

        if query_head_count == key_value_head_count:
            serialize_f32(
                output_file,
                unpermute(
                    attention_key_weight_matrix,
                    query_head_count,
                    embedding_size,
                    embedding_size,
                ),
            )
        else:
            serialize_f32(
                output_file,
                unpermute(
                    attention_key_weight_matrix,
                    key_value_head_count,
                    key_value_size,
                    embedding_size,
                ),
            )

        attention_value_weight_matrix = state[
            f"model.layers.{layer}.self_attn.v_proj.weight"
        ]

        serialize_f32(output_file, attention_value_weight_matrix)

        attention_output_weight_matrix = state[
            f"model.layers.{layer}.self_attn.o_proj.weight"
        ]

        serialize_f32(output_file, attention_output_weight_matrix)

    # FNN layers ###################################################################################

    for layer in range(layer_count):
        fnn_norm_weight_vector = state[
            f"model.layers.{layer}.post_attention_layernorm.weight"
        ]

        serialize_f32(output_file, fnn_norm_weight_vector)

        fnn_gate_weight_matrix = state[f"model.layers.{layer}.mlp.gate_proj.weight"]

        serialize_f32(output_file, fnn_gate_weight_matrix)

        fnn_up_weight_matrix = state[f"model.layers.{layer}.mlp.up_proj.weight"]

        serialize_f32(output_file, fnn_up_weight_matrix)

        fnn_down_weight_matrix = state[f"model.layers.{layer}.mlp.down_proj.weight"]

        serialize_f32(output_file, fnn_down_weight_matrix)

    # Linear layer #################################################################################

    serialize_f32(output_file, linear_norm_weight_vector)

    if not shared_output_weight:
        serialize_f32(output_file, linear_output_weight_matrix)

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_model_path", type=str, help="the input model")
    parser.add_argument("output_model_path", type=str, help="the output model")

    args = parser.parse_args()

    write_model_file()
