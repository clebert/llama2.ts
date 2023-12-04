const std = @import("std");

const addition = @import("simd/addition.zig");
const dot_product = @import("simd/dot_product.zig");
const matrix_vector_product = @import("simd/matrix_vector_product.zig");
const rms_norm = @import("simd/rms_norm.zig");
const softmax = @import("simd/softmax.zig");

const Self = @This();

num_layers: usize,
num_query_heads: usize,
num_kv_heads: usize,
norm_weight: []f32,
query_weight: []f32,
key_weight: []f32,
value_weight: []f32,
output_weight: []f32,
input_vector: []f32,
output_vector: []f32,

// private
attention_vector: []f32,
query_vector: []f32,
key_cache: []f32,
value_cache: []f32,
scores: []f32,

export fn create(
    query_size: usize,
    max_sequence_len: usize,
    num_layers: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
) ?*Self {
    const allocator = std.heap.page_allocator;
    const self = allocator.create(Self) catch return null;
    const kv_size = num_kv_heads * (query_size / num_query_heads);

    self.* = .{
        .num_layers = num_layers,
        .num_query_heads = num_query_heads,
        .num_kv_heads = num_kv_heads,
        .norm_weight = allocator.alloc(f32, num_layers * query_size) catch return null,
        .query_weight = allocator.alloc(f32, num_layers * query_size * query_size) catch return null,
        .key_weight = allocator.alloc(f32, num_layers * kv_size * query_size) catch return null,
        .value_weight = allocator.alloc(f32, num_layers * kv_size * query_size) catch return null,
        .output_weight = allocator.alloc(f32, num_layers * query_size * query_size) catch return null,
        .input_vector = allocator.alloc(f32, query_size) catch return null,
        .output_vector = allocator.alloc(f32, query_size) catch return null,
        .attention_vector = allocator.alloc(f32, query_size) catch return null,
        .query_vector = allocator.alloc(f32, query_size) catch return null,
        .key_cache = allocator.alloc(f32, max_sequence_len * num_layers * kv_size) catch return null,
        .value_cache = allocator.alloc(f32, max_sequence_len * num_layers * kv_size) catch return null,
        .scores = allocator.alloc(f32, max_sequence_len) catch return null,
    };

    return self;
}

export fn getNormWeight(self: *const Self) [*]f32 {
    return self.norm_weight.ptr;
}

export fn getQueryWeight(self: *const Self) [*]f32 {
    return self.query_weight.ptr;
}

export fn getKeyWeight(self: *const Self) [*]f32 {
    return self.key_weight.ptr;
}

export fn getValueWeight(self: *const Self) [*]f32 {
    return self.value_weight.ptr;
}

export fn getOutputWeight(self: *const Self) [*]f32 {
    return self.output_weight.ptr;
}

export fn getInputVector(self: *const Self) [*]f32 {
    return self.input_vector.ptr;
}

export fn getOutputVector(self: *const Self) [*]f32 {
    return self.output_vector.ptr;
}

export fn forward(self: *const Self, position: usize, layer: usize) void {
    @setFloatMode(.Optimized);

    rms_norm.compute(
        self.norm_weight[layer * self.input_vector.len ..][0..self.input_vector.len],
        self.input_vector,
        self.attention_vector,
    );

    const query_weight_size = self.query_vector.len * self.attention_vector.len;

    matrix_vector_product.compute(
        self.query_weight[layer * query_weight_size ..][0..query_weight_size],
        self.attention_vector,
        self.query_vector,
    );

    const head_size = self.query_vector.len / self.num_query_heads;
    const kv_size = self.num_kv_heads * head_size;
    const kv_weight_size = kv_size * self.attention_vector.len;
    const kv_cache_offset = (position * self.num_layers + layer) * kv_size;
    const key_vector = self.key_cache[kv_cache_offset..][0..kv_size];

    matrix_vector_product.compute(
        self.key_weight[layer * kv_weight_size ..][0..kv_weight_size],
        self.attention_vector,
        key_vector,
    );

    computeRope(self, head_size, key_vector, position);

    matrix_vector_product.compute(
        self.value_weight[layer * kv_weight_size ..][0..kv_weight_size],
        self.attention_vector,
        self.value_cache[kv_cache_offset..][0..kv_size],
    );

    const scores = self.scores[0 .. position + 1];
    const head_size_sqrt = @sqrt(@as(f32, @floatFromInt(head_size)));

    // Grouped-query attention: https://arxiv.org/abs/2305.13245v1
    for (0..self.num_query_heads) |query_head| {
        const query_head_offset = query_head * head_size;
        const query_head_vector = self.query_vector[query_head_offset..][0..head_size];

        const kv_head = query_head / (self.num_query_heads / self.num_kv_heads);
        const kv_head_offset = kv_head * head_size;

        for (scores, 0..) |*score, prev_position| {
            const prev_kv_cache_offset = (prev_position * self.num_layers + layer) * kv_size;
            const prev_key_vector = self.key_cache[prev_kv_cache_offset..][0..kv_size];
            const prev_key_head_vector = prev_key_vector[kv_head_offset..][0..head_size];

            score.* = dot_product.compute(query_head_vector, prev_key_head_vector) / head_size_sqrt;
        }

        softmax.compute(scores, scores);

        const attention_head_vector = self.attention_vector[query_head_offset..][0..head_size];

        @memset(attention_head_vector, 0);

        for (scores, 0..) |score, prev_position| {
            const prev_kv_cache_offset = (prev_position * self.num_layers + layer) * kv_size;
            const prev_value_vector = self.value_cache[prev_kv_cache_offset..][0..kv_size];
            const prev_value_head_vector = prev_value_vector[kv_head_offset..][0..head_size];

            for (0..attention_head_vector.len) |index| {
                attention_head_vector[index] += prev_value_head_vector[index] * score;
            }
        }
    }

    matrix_vector_product.compute(
        self.output_weight[layer * query_weight_size ..][0..query_weight_size],
        self.attention_vector,
        self.output_vector,
    );

    addition.compute(self.output_vector, self.input_vector, self.output_vector);
}

// Rotary positional embeddings: https://arxiv.org/abs/2104.09864
fn computeRope(self: *const Self, head_size: usize, key_vector: []f32, position: usize) void {
    @setFloatMode(.Optimized);

    var index: usize = 0;

    while (index < self.query_vector.len) : (index += 2) {
        const inverse_frequency = 1 / std.math.pow(
            f32,
            10000,
            @as(f32, @floatFromInt(index % head_size)) / @as(f32, @floatFromInt(head_size)),
        );

        const theta: f32 = @as(f32, @floatFromInt(position)) * inverse_frequency;
        const real_rotation: f32 = std.math.cos(theta);
        const imag_rotation: f32 = std.math.sin(theta);

        const q_0 = self.query_vector[index];
        const q_1 = self.query_vector[index + 1];

        self.query_vector[index] = q_0 * real_rotation - q_1 * imag_rotation;
        self.query_vector[index + 1] = q_0 * imag_rotation + q_1 * real_rotation;

        if (index < key_vector.len) {
            const k_0 = key_vector[index];
            const k_1 = key_vector[index + 1];

            key_vector[index] = k_0 * real_rotation - k_1 * imag_rotation;
            key_vector[index + 1] = k_0 * imag_rotation + k_1 * real_rotation;
        }
    }
}
