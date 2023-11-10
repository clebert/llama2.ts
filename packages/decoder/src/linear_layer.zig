const std = @import("std");
const matrix_vector_product = @import("simd/matrix_vector_product.zig");
const rms_norm = @import("simd/rms_norm.zig");

const State = struct {
    input_vector: []f32,
    norm_weight_vector: []f32,
    output_weight_matrix: []f32,
    output_vector: []f32,
};

export fn init(embedding_size: usize, vocab_size: usize) ?*const State {
    const allocator = std.heap.page_allocator;
    const state = allocator.create(State) catch return null;

    state.* = .{
        .input_vector = allocator.alloc(f32, embedding_size) catch return null,
        .norm_weight_vector = allocator.alloc(f32, embedding_size) catch return null,
        .output_weight_matrix = allocator.alloc(f32, vocab_size * embedding_size) catch return null,
        .output_vector = allocator.alloc(f32, vocab_size) catch return null,
    };

    return state;
}

export fn getInputVector(state: *const State) [*]f32 {
    return state.input_vector.ptr;
}

export fn getNormWeightVector(state: *const State) [*]f32 {
    return state.norm_weight_vector.ptr;
}

export fn getOutputWeightMatrix(state: *const State) [*]f32 {
    return state.output_weight_matrix.ptr;
}

export fn getOutputVector(state: *const State) [*]f32 {
    return state.output_vector.ptr;
}

export fn forward(state: *const State) void {
    rms_norm.compute(state.input_vector, state.norm_weight_vector, state.input_vector);

    matrix_vector_product.compute(
        state.output_weight_matrix,
        state.input_vector,
        state.output_vector,
    );
}
