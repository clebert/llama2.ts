const std = @import("std");

const exp = @import("exp.zig");
const scalar_divison = @import("scalar_divison.zig");
const scalar_subtraction = @import("scalar_subtraction.zig");
const reduce_add = @import("reduce_add.zig");
const reduce_max = @import("reduce_max.zig");

pub fn compute(input_vector: []const f32, output_vector: []f32) void {
    scalar_subtraction.compute(input_vector, reduce_max.compute(input_vector), output_vector);
    exp.compute(output_vector, output_vector);
    scalar_divison.compute(output_vector, reduce_add.compute(output_vector), output_vector);
}

test "compute softmax" {
    const allocator = std.testing.allocator;
    const input_vector = [_]f32{ 0.1, 4.5, -0.2, 3.3, 5.4 };
    const output_vector = try allocator.alloc(f32, 5);

    defer allocator.free(output_vector);

    compute(&input_vector, output_vector);

    try std.testing.expectApproxEqRel(output_vector[0], 0.00324611, std.math.floatEps(f32));
    try std.testing.expectApproxEqRel(output_vector[1], 0.264398485, std.math.floatEps(f32));
    try std.testing.expectApproxEqRel(output_vector[2], 0.00240477779, std.math.floatEps(f32));
    try std.testing.expectApproxEqRel(output_vector[3], 0.0796352848, std.math.floatEps(f32));
    try std.testing.expectApproxEqRel(output_vector[4], 0.650315403, std.math.floatEps(f32));
    try std.testing.expectApproxEqRel(reduce_add.compute(output_vector), 1, std.math.floatEps(f32));
}
