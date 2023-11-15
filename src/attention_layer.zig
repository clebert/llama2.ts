const std = @import("std");
const dot_product = @import("simd/dot_product.zig");
const matrix_vector_product = @import("simd/matrix_vector_product.zig");
const rms_norm = @import("simd/rms_norm.zig");
const softmax = @import("simd/softmax.zig");

const State = struct {
    query_head_count: usize,
    key_value_head_count: usize,
    head_size: usize,
    head_size_sqrt: f32,

    input_vector: []f32,
    norm_weight_vector: []f32,
    query_weight_matrix: []f32,
    key_weight_matrix: []f32,
    value_weight_matrix: []f32,
    output_weight_matrix: []f32,
    output_vector: []f32,

    query_vector: []f32,
    key_vectors: [][]f32,
    value_vectors: [][]f32,
    scores: []f32,
};

export fn init(
    embedding_size: usize,
    key_value_size: usize,
    query_head_count: usize,
    max_sequence_len: usize,
) ?*const State {
    const allocator = std.heap.page_allocator;
    const head_size = embedding_size / query_head_count;
    const state = allocator.create(State) catch return null;
    const key_vectors = allocator.alloc([]f32, max_sequence_len) catch return null;
    const value_vectors = allocator.alloc([]f32, max_sequence_len) catch return null;

    for (key_vectors) |*key_vector| {
        key_vector.* = allocator.alloc(f32, key_value_size) catch return null;
    }

    for (value_vectors) |*value_vector| {
        value_vector.* = allocator.alloc(f32, key_value_size) catch return null;
    }

    state.* = .{
        .query_head_count = query_head_count,
        .key_value_head_count = key_value_size / head_size,
        .head_size = head_size,
        .head_size_sqrt = std.math.sqrt(@as(f32, @floatFromInt(head_size))),

        .input_vector = allocator.alloc(f32, embedding_size) catch return null,
        .norm_weight_vector = allocator.alloc(f32, embedding_size) catch return null,

        .query_weight_matrix = allocator.alloc(
            f32,
            embedding_size * embedding_size,
        ) catch return null,

        .key_weight_matrix = allocator.alloc(
            f32,
            key_value_size * embedding_size,
        ) catch return null,

        .value_weight_matrix = allocator.alloc(
            f32,
            key_value_size * embedding_size,
        ) catch return null,

        .output_weight_matrix = allocator.alloc(
            f32,
            embedding_size * embedding_size,
        ) catch return null,

        .output_vector = allocator.alloc(f32, embedding_size) catch return null,

        .query_vector = allocator.alloc(f32, embedding_size) catch return null,
        .key_vectors = key_vectors,
        .value_vectors = value_vectors,
        .scores = allocator.alloc(f32, max_sequence_len) catch return null,
    };

    return state;
}

export fn getInputVector(state: *const State) [*]f32 {
    return state.input_vector.ptr;
}

export fn getNormWeightVector(state: *const State) [*]f32 {
    return state.norm_weight_vector.ptr;
}

export fn getQueryWeightMatrix(state: *const State) [*]f32 {
    return state.query_weight_matrix.ptr;
}

export fn getKeyWeightMatrix(state: *const State) [*]f32 {
    return state.key_weight_matrix.ptr;
}

export fn getValueWeightMatrix(state: *const State) [*]f32 {
    return state.value_weight_matrix.ptr;
}

export fn getOutputWeightMatrix(state: *const State) [*]f32 {
    return state.output_weight_matrix.ptr;
}

export fn getOutputVector(state: *const State) [*]f32 {
    return state.output_vector.ptr;
}

export fn forward(state: *const State, position: usize) void {
    rms_norm.compute(state.norm_weight_vector, state.input_vector, state.input_vector);

    matrix_vector_product.compute(
        state.query_weight_matrix,
        state.input_vector,
        state.query_vector,
    );

    const key_vector = state.key_vectors[position];
    const value_vector = state.value_vectors[position];

    matrix_vector_product.compute(state.key_weight_matrix, state.input_vector, key_vector);
    matrix_vector_product.compute(state.value_weight_matrix, state.input_vector, value_vector);

    computeRope(state, position);

    for (0..state.query_head_count) |query_head| {
        computeGQA(state, position, query_head);
    }

    matrix_vector_product.compute(
        state.output_weight_matrix,
        state.input_vector,
        state.output_vector,
    );
}

// Rotary positional embeddings: https://arxiv.org/abs/2104.09864
fn computeRope(state: *const State, position: usize) void {
    @setFloatMode(.Optimized);

    const key_vector = state.key_vectors[position];

    var index: usize = 0;

    while (index < state.query_vector.len) : (index += 2) {
        const inverse_frequency = 1 / std.math.pow(
            f32,
            10000,
            @as(f32, @floatFromInt(index % state.head_size)) /
                @as(f32, @floatFromInt(state.head_size)),
        );

        const theta: f32 = @as(f32, @floatFromInt(position)) * inverse_frequency;
        const real_rotation: f32 = std.math.cos(theta);
        const imag_rotation: f32 = std.math.sin(theta);

        const q_0 = state.query_vector[index];
        const q_1 = state.query_vector[index + 1];

        state.query_vector[index] = q_0 * real_rotation - q_1 * imag_rotation;
        state.query_vector[index + 1] = q_0 * imag_rotation + q_1 * real_rotation;

        if (index < key_vector.len) {
            const k_0 = key_vector[index];
            const k_1 = key_vector[index + 1];

            key_vector[index] = k_0 * real_rotation - k_1 * imag_rotation;
            key_vector[index + 1] = k_0 * imag_rotation + k_1 * real_rotation;
        }
    }
}

// Grouped-query attention: https://arxiv.org/abs/2305.13245v1
fn computeGQA(state: *const State, position: usize, query_head: usize) void {
    @setFloatMode(.Optimized);

    const query_head_index = query_head * state.head_size;
    const query_head_vector = state.query_vector[query_head_index..][0..state.head_size];

    const key_value_head = query_head / (state.query_head_count / state.key_value_head_count);
    const key_value_head_index = key_value_head * state.head_size;

    for (0..position + 1) |other_position| {
        const key_vector = state.key_vectors[other_position];
        const key_head_vector = key_vector[key_value_head_index..][0..state.head_size];

        state.scores[other_position] =
            dot_product.compute(query_head_vector, key_head_vector) / state.head_size_sqrt;
    }

    const scores = state.scores[0 .. position + 1];

    softmax.compute(scores, scores);

    const input_head_vector = state.input_vector[query_head_index..][0..state.head_size];

    @memset(input_head_vector, 0);

    for (0..position + 1) |other_position| {
        const value_vector = state.value_vectors[other_position];
        const value_head_vector = value_vector[key_value_head_index..][0..state.head_size];
        const weight = scores[other_position];

        for (0..state.head_size) |index| {
            input_head_vector[index] += value_head_vector[index] * weight;
        }
    }
}
