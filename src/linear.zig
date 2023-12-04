const std = @import("std");

const matrix_vector_product = @import("simd/matrix_vector_product.zig");
const rms_norm = @import("simd/rms_norm.zig");
const softmax = @import("simd/softmax.zig");

const Self = @This();

norm_weight: []f32,
output_weight: []f32,
input_vector: []f32,
output_vector: []f32,

// private
norm_vector: []f32,

export fn create(input_size: usize, output_size: usize) ?*Self {
    const allocator = std.heap.page_allocator;
    const self = allocator.create(Self) catch return null;

    self.* = .{
        .norm_weight = allocator.alloc(f32, input_size) catch return null,
        .output_weight = allocator.alloc(f32, output_size * input_size) catch return null,
        .input_vector = allocator.alloc(f32, input_size) catch return null,
        .output_vector = allocator.alloc(f32, output_size) catch return null,
        .norm_vector = allocator.alloc(f32, input_size) catch return null,
    };

    return self;
}

export fn getNormWeight(self: *const Self) [*]f32 {
    return self.norm_weight.ptr;
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

export fn forward(self: *const Self, compute_softmax: bool) void {
    rms_norm.compute(self.norm_weight, self.input_vector, self.norm_vector);
    matrix_vector_product.compute(self.output_weight, self.norm_vector, self.output_vector);

    if (compute_softmax) {
        softmax.compute(self.output_vector, self.output_vector);
    }
}
