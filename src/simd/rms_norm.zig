const std = @import("std");

const dot_product = @import("dot_product.zig");
const hadamard_product = @import("hadamard_product.zig");
const scalar_multiplication = @import("scalar_multiplication.zig");

// Pre-normalization using RMSNorm: https://arxiv.org/abs/1910.07467
pub fn compute(weight_vector: []const f32, input_vector: []const f32, output_vector: []f32) void {
    @setFloatMode(.Optimized);

    var scaling_factor = dot_product.compute(input_vector, input_vector);

    scaling_factor /= @floatFromInt(input_vector.len);
    scaling_factor += 1e-5;
    scaling_factor = 1 / std.math.sqrt(scaling_factor);

    scalar_multiplication.compute(input_vector, scaling_factor, output_vector);
    hadamard_product.compute(weight_vector, output_vector, output_vector);
}

test "rms_norm" {
    const allocator = std.testing.allocator;
    const weight_vector = [_]f32{ 5, 6, 7, 8 };
    const input_vector = [_]f32{ 1, 2, 3, 4 };
    const output_vector = try allocator.alloc(f32, 4);

    defer allocator.free(output_vector);

    compute(&weight_vector, &input_vector, output_vector);

    try std.testing.expectApproxEqRel(output_vector[0], 1.82574057, std.math.floatEps(f32));
    try std.testing.expectApproxEqRel(output_vector[1], 4.38177776, std.math.floatEps(f32));
    try std.testing.expectApproxEqRel(output_vector[2], 7.66811084, std.math.floatEps(f32));
    try std.testing.expectApproxEqRel(output_vector[3], 11.68474, std.math.floatEps(f32));
}
