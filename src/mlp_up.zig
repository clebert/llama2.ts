const std = @import("std");

const addition = @import("simd/addition.zig");
const hadamard_product = @import("simd/hadamard_product.zig");
const matrix_vector_product = @import("simd/matrix_vector_product.zig");
const rms_norm = @import("simd/rms_norm.zig");
const swish = @import("simd/swish.zig");

const Self = @This();

norm_weight: []f32,
gate_weight: []f32,
up_weight: []f32,
input_vector: []f32,
output_vector: []f32,

// private
norm_vector: []f32,
gate_vector: []f32,

export fn create(input_size: usize, output_size: usize, num_layers: usize) ?*Self {
    const allocator = std.heap.page_allocator;
    const self = allocator.create(Self) catch return null;

    self.* = .{
        .norm_weight = allocator.alloc(f32, num_layers * input_size) catch return null,
        .gate_weight = allocator.alloc(f32, num_layers * output_size * input_size) catch return null,
        .up_weight = allocator.alloc(f32, num_layers * output_size * input_size) catch return null,
        .input_vector = allocator.alloc(f32, input_size) catch return null,
        .output_vector = allocator.alloc(f32, output_size) catch return null,
        .norm_vector = allocator.alloc(f32, input_size) catch return null,
        .gate_vector = allocator.alloc(f32, output_size) catch return null,
    };

    return self;
}

export fn getNormWeight(self: *const Self) [*]f32 {
    return self.norm_weight.ptr;
}

export fn getGateWeight(self: *const Self) [*]f32 {
    return self.gate_weight.ptr;
}

export fn getUpWeight(self: *const Self) [*]f32 {
    return self.up_weight.ptr;
}

export fn getInputVector(self: *const Self) [*]f32 {
    return self.input_vector.ptr;
}

export fn getOutputVector(self: *const Self) [*]f32 {
    return self.output_vector.ptr;
}

export fn forward(self: *const Self, layer: usize) void {
    const norm_weight_size = self.input_vector.len;

    rms_norm.compute(
        self.norm_weight[layer * norm_weight_size ..][0..norm_weight_size],
        self.input_vector,
        self.norm_vector,
    );

    const up_weight_size = self.output_vector.len * self.input_vector.len;

    matrix_vector_product.compute(
        self.gate_weight[layer * up_weight_size ..][0..up_weight_size],
        self.norm_vector,
        self.gate_vector,
    );

    swish.compute(self.gate_vector, self.gate_vector);

    matrix_vector_product.compute(
        self.up_weight[layer * up_weight_size ..][0..up_weight_size],
        self.norm_vector,
        self.output_vector,
    );

    hadamard_product.compute(self.output_vector, self.gate_vector, self.output_vector);
}
