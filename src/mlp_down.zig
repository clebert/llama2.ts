const std = @import("std");

const addition = @import("simd/addition.zig");
const hadamard_product = @import("simd/hadamard_product.zig");
const matrix_vector_product = @import("simd/matrix_vector_product.zig");
const rms_norm = @import("simd/rms_norm.zig");
const swish = @import("simd/swish.zig");

const Self = @This();

down_weight: []f32,
input_vector: []f32,
output_vector: []f32,
residual_vector: []f32,

export fn create(input_size: usize, output_size: usize, num_layers: usize) ?*Self {
    const allocator = std.heap.page_allocator;
    const self = allocator.create(Self) catch return null;

    self.* = .{
        .down_weight = allocator.alloc(f32, num_layers * output_size * input_size) catch return null,
        .input_vector = allocator.alloc(f32, input_size) catch return null,
        .output_vector = allocator.alloc(f32, output_size) catch return null,
        .residual_vector = allocator.alloc(f32, output_size) catch return null,
    };

    return self;
}

export fn getDownWeight(self: *const Self) [*]f32 {
    return self.down_weight.ptr;
}

export fn getInputVector(self: *const Self) [*]f32 {
    return self.input_vector.ptr;
}

export fn getOutputVector(self: *const Self) [*]f32 {
    return self.output_vector.ptr;
}

export fn getResidualVector(self: *const Self) [*]f32 {
    return self.residual_vector.ptr;
}

export fn forward(self: *const Self, layer: usize) void {
    const weight_size = self.output_vector.len * self.input_vector.len;

    matrix_vector_product.compute(
        self.down_weight[layer * weight_size ..][0..weight_size],
        self.input_vector,
        self.output_vector,
    );

    addition.compute(self.output_vector, self.residual_vector, self.output_vector);
}
