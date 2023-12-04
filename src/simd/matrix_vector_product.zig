const std = @import("std");

const dot_product = @import("dot_product.zig");

pub fn compute(matrix: []const f32, input_vector: []const f32, output_vector: []f32) void {
    std.debug.assert(matrix.len == output_vector.len * input_vector.len);
    std.debug.assert(@intFromPtr(input_vector.ptr) != @intFromPtr(output_vector.ptr));

    for (output_vector, 0..) |*output, index| {
        output.* = dot_product.compute(
            matrix[index * input_vector.len ..][0..input_vector.len],
            input_vector,
        );
    }
}

test "matrix_vector_product" {
    const allocator = std.testing.allocator;
    const matrix = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const input_vector = [_]f32{ 7, 8 };
    const output_vector = try allocator.alloc(f32, 3);

    defer allocator.free(output_vector);

    compute(&matrix, &input_vector, output_vector);

    try std.testing.expectApproxEqRel(output_vector[0], 1 * 7 + 2 * 8, std.math.floatEps(f32));
    try std.testing.expectApproxEqRel(output_vector[1], 3 * 7 + 4 * 8, std.math.floatEps(f32));
    try std.testing.expectApproxEqRel(output_vector[2], 5 * 7 + 6 * 8, std.math.floatEps(f32));
}
