const std = @import("std");

const constants = @import("constants.zig");

pub fn compute(input_vector: []const f32) f32 {
    comptime var vector_len = constants.max_vector_len;

    var output: f32 = -std.math.floatMax(f32);
    var rest_input_vector = input_vector;

    inline while (vector_len >= constants.min_vector_len) : (vector_len /= 2) {
        if (rest_input_vector.len >= vector_len) {
            computeFixed(vector_len, rest_input_vector, &output);

            rest_input_vector =
                input_vector[input_vector.len - (rest_input_vector.len % vector_len) ..];
        }
    }

    if (rest_input_vector.len > 0) {
        computeFixed(1, rest_input_vector, &output);
    }

    return output;
}

fn computeFixed(comptime vector_len: comptime_int, input_vector: []const f32, output: *f32) void {
    @setFloatMode(.Optimized);

    var offset: usize = 0;

    while (offset + vector_len <= input_vector.len) : (offset += vector_len) {
        output.* = @max(
            output.*,
            @reduce(.Max, @as(@Vector(vector_len, f32), input_vector[offset..][0..vector_len].*)),
        );
    }
}

test "reduce_max" {
    const allocator = std.testing.allocator;

    for (0..2) |vector_len_subtrahend| {
        const vector_len = constants.max_vector_len - vector_len_subtrahend;
        const input_vector = try allocator.alloc(f32, vector_len);

        defer allocator.free(input_vector);

        for (0..input_vector.len) |index| {
            input_vector[index] = @floatFromInt(index + 1);
        }

        try std.testing.expectApproxEqRel(
            @as(f32, @floatFromInt(input_vector.len)),
            compute(input_vector),
            std.math.floatEps(f32),
        );
    }
}

test "reduce_max (reverse)" {
    const allocator = std.testing.allocator;

    for (0..2) |vector_len_subtrahend| {
        const vector_len = constants.max_vector_len - vector_len_subtrahend;
        const input_vector = try allocator.alloc(f32, vector_len);

        defer allocator.free(input_vector);

        for (0..input_vector.len) |index| {
            input_vector[index] = @floatFromInt(input_vector.len - index);
        }

        try std.testing.expectApproxEqRel(
            @as(f32, @floatFromInt(input_vector.len)),
            compute(input_vector),
            std.math.floatEps(f32),
        );
    }
}
