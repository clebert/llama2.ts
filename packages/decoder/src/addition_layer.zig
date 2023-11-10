const std = @import("std");

const addition = @import("simd/addition.zig");

const State = struct {
    input_vector: []f32,
    output_vector: []f32,
};

export fn init(embedding_size: usize) ?*const State {
    const allocator = std.heap.page_allocator;
    const state = allocator.create(State) catch return null;

    state.* = .{
        .input_vector = allocator.alloc(f32, embedding_size) catch return null,
        .output_vector = allocator.alloc(f32, embedding_size) catch return null,
    };

    return state;
}

export fn getInputVector(state: *const State) [*]f32 {
    return state.input_vector.ptr;
}

export fn getOutputVector(state: *const State) [*]f32 {
    return state.output_vector.ptr;
}

export fn forward(state: *const State) void {
    addition.compute(state.input_vector, state.output_vector, state.output_vector);
}
