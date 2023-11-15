const std = @import("std");

const addition = @import("simd/addition.zig");
const dot_product = @import("simd/dot_product.zig");
const hadamard_product = @import("simd/hadamard_product.zig");
const rms_norm = @import("simd/rms_norm.zig");
const softmax = @import("simd/softmax.zig");
const swish = @import("simd/swish.zig");

const State = struct {
    vector: []f32,
    input_vector: []f32,
    output_vector: []f32,
};

export fn init(vector_len: usize) ?*const State {
    const allocator = std.heap.page_allocator;
    const state = allocator.create(State) catch return null;

    state.* = .{
        .vector = allocator.alloc(f32, vector_len) catch return null,
        .input_vector = allocator.alloc(f32, vector_len) catch return null,
        .output_vector = allocator.alloc(f32, vector_len) catch return null,
    };

    return state;
}

export fn getVector(state: *const State) [*]f32 {
    return state.vector.ptr;
}

export fn getInputVector(state: *const State) [*]f32 {
    return state.input_vector.ptr;
}

export fn getOutputVector(state: *const State) [*]f32 {
    return state.output_vector.ptr;
}

// TODO: computeArgmax

export fn computeAddition(state: *const State) void {
    addition.compute(state.vector, state.input_vector, state.output_vector);
}

export fn computeDotProduct(state: *const State) f32 {
    return dot_product.compute(state.vector, state.input_vector);
}

export fn computeHadamardProduct(state: *const State) void {
    hadamard_product.compute(state.vector, state.input_vector, state.output_vector);
}

export fn computeRmsNorm(state: *const State) void {
    rms_norm.compute(state.vector, state.input_vector, state.output_vector);
}

export fn computeSoftmax(state: *const State) void {
    softmax.compute(state.vector, state.output_vector);
}

export fn computeSwish(state: *const State) void {
    swish.compute(state.vector, state.output_vector);
}
