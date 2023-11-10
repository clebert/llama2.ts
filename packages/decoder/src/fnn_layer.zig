const std = @import("std");
const hadamard_product = @import("simd/hadamard_product.zig");
const matrix_vector_product = @import("simd/matrix_vector_product.zig");
const rms_norm = @import("simd/rms_norm.zig");
const swish = @import("simd/swish.zig");

const State = struct {
    input_vector: []f32,
    norm_weight_vector: []f32,
    gate_weight_matrix: []f32,
    up_weight_matrix: []f32,
    down_weight_matrix: []f32,
    output_vector: []f32,

    hidden_vector_1: []f32,
    hidden_vector_2: []f32,
};

export fn init(embedding_size: usize, hidden_size: usize) ?*const State {
    const allocator = std.heap.page_allocator;
    const state = allocator.create(State) catch return null;

    state.* = .{
        .input_vector = allocator.alloc(f32, embedding_size) catch return null,
        .norm_weight_vector = allocator.alloc(f32, embedding_size) catch return null,
        .gate_weight_matrix = allocator.alloc(f32, hidden_size * embedding_size) catch return null,
        .up_weight_matrix = allocator.alloc(f32, hidden_size * embedding_size) catch return null,
        .down_weight_matrix = allocator.alloc(f32, embedding_size * hidden_size) catch return null,
        .output_vector = allocator.alloc(f32, embedding_size) catch return null,

        .hidden_vector_1 = allocator.alloc(f32, hidden_size) catch return null,
        .hidden_vector_2 = allocator.alloc(f32, hidden_size) catch return null,
    };

    return state;
}

export fn getInputVector(state: *const State) [*]f32 {
    return state.input_vector.ptr;
}

export fn getNormWeightVector(state: *const State) [*]f32 {
    return state.norm_weight_vector.ptr;
}

export fn getGateWeightMatrix(state: *const State) [*]f32 {
    return state.gate_weight_matrix.ptr;
}

export fn getUpWeightMatrix(state: *const State) [*]f32 {
    return state.up_weight_matrix.ptr;
}

export fn getDownWeightMatrix(state: *const State) [*]f32 {
    return state.down_weight_matrix.ptr;
}

export fn getOutputVector(state: *const State) [*]f32 {
    return state.output_vector.ptr;
}

// SwiGLU activation function: https://arxiv.org/abs/2002.05202
export fn forward(state: *const State) void {
    rms_norm.compute(state.input_vector, state.norm_weight_vector, state.input_vector);

    matrix_vector_product.compute(
        state.gate_weight_matrix,
        state.input_vector,
        state.hidden_vector_1,
    );

    matrix_vector_product.compute(
        state.up_weight_matrix,
        state.input_vector,
        state.hidden_vector_2,
    );

    swish.compute(state.hidden_vector_1, state.hidden_vector_1);
    hadamard_product.compute(state.hidden_vector_1, state.hidden_vector_2, state.hidden_vector_1);

    matrix_vector_product.compute(
        state.down_weight_matrix,
        state.hidden_vector_1,
        state.output_vector,
    );
}
