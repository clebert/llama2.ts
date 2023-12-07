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
def unpermute(tensor, num_query_heads, dim_1, dim_2):
    return (
        tensor.view(num_query_heads, 2, dim_1 // num_query_heads // 2, dim_2)
        .transpose(1, 2)
        .reshape(dim_1, dim_2)
    )


def write_model_file():
    model = AutoModelForCausalLM.from_pretrained(args.input_model_path)

    if model.config.model_type != "llama":
        parser.error("Expected Llama model")

    if model.config.rope_theta != 10000:
        parser.error("Expected RoPE frequency base of 10000")

    if model.config.rms_norm_eps != 1e-05:
        parser.error("Expected RMS norm eps of 1e-05")

    state = model.state_dict()

    embedding_weight = state["model.embed_tokens.weight"]
    linear_norm_weight = state["model.norm.weight"]
    linear_output_weight = state[f"lm_head.weight"]

    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    max_sequence_length = model.config.max_position_embeddings
    vocab_size = model.config.vocab_size
    num_layers = model.config.num_hidden_layers
    num_query_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    shared_output_weight = torch.equal(embedding_weight, linear_output_weight)

    key_value_size = num_key_value_heads * (hidden_size // num_query_heads)

    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)

    output_file = open(args.output_model_path, "wb")

    # Header #######################################################################################

    output_file.write(struct.pack("B", 1))
    output_file.write(struct.pack("i", 5))
    output_file.write("llama".encode("utf-8"))

    output_file.write(
        struct.pack(
            "iiiiiii",
            hidden_size,
            intermediate_size,
            max_sequence_length,
            vocab_size,
            num_layers,
            num_query_heads,
            num_key_value_heads,
        )
    )

    output_file.write(struct.pack("B", int(shared_output_weight)))

    output_file.write(b"\0" * (256 - output_file.tell()))

    # Vocab ########################################################################################

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

    # Checkpoint ###################################################################################

    serialize_f32(output_file, embedding_weight)

    for layer in range(num_layers):
        attention_norm_weight = state[f"model.layers.{layer}.input_layernorm.weight"]

        serialize_f32(output_file, attention_norm_weight)

    for layer in range(num_layers):
        attention_query_weight = state[f"model.layers.{layer}.self_attn.q_proj.weight"]

        serialize_f32(
            output_file,
            unpermute(
                attention_query_weight, num_query_heads, hidden_size, hidden_size
            ),
        )

    for layer in range(num_layers):
        attention_key_weight = state[f"model.layers.{layer}.self_attn.k_proj.weight"]

        if num_query_heads == num_key_value_heads:
            serialize_f32(
                output_file,
                unpermute(
                    attention_key_weight, num_query_heads, hidden_size, hidden_size
                ),
            )
        else:
            serialize_f32(
                output_file,
                unpermute(
                    attention_key_weight,
                    num_key_value_heads,
                    key_value_size,
                    hidden_size,
                ),
            )

    for layer in range(num_layers):
        attention_value_weight = state[f"model.layers.{layer}.self_attn.v_proj.weight"]

        serialize_f32(output_file, attention_value_weight)

    for layer in range(num_layers):
        attention_output_weight = state[f"model.layers.{layer}.self_attn.o_proj.weight"]

        serialize_f32(output_file, attention_output_weight)

    for layer in range(num_layers):
        mlp_norm_weight = state[f"model.layers.{layer}.post_attention_layernorm.weight"]

        serialize_f32(output_file, mlp_norm_weight)

    for layer in range(num_layers):
        mlp_gate_weight = state[f"model.layers.{layer}.mlp.gate_proj.weight"]

        serialize_f32(output_file, mlp_gate_weight)

    for layer in range(num_layers):
        mlp_up_weight = state[f"model.layers.{layer}.mlp.up_proj.weight"]

        serialize_f32(output_file, mlp_up_weight)

    for layer in range(num_layers):
        mlp_down_weight = state[f"model.layers.{layer}.mlp.down_proj.weight"]

        serialize_f32(output_file, mlp_down_weight)

    serialize_f32(output_file, linear_norm_weight)

    if not shared_output_weight:
        serialize_f32(output_file, linear_output_weight)

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_model_path", type=str, help="the input model")
    parser.add_argument("output_model_path", type=str, help="the output model")

    args = parser.parse_args()

    write_model_file()
