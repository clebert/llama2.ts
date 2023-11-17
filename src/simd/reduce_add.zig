const std = @import("std");

const constants = @import("constants.zig");

pub fn compute(input_vector: []const f32) f32 {
    comptime var vector_len = constants.max_vector_len;

    var output: f32 = 0;
    var rest_input_vector = input_vector;

    inline while (vector_len >= constants.min_vector_len) : (vector_len /= 2) {
        if (rest_input_vector.len >= vector_len) {
            output += computeFixed(vector_len, rest_input_vector);

            rest_input_vector =
                input_vector[input_vector.len - (rest_input_vector.len % vector_len) ..];
        }
    }

    if (rest_input_vector.len > 0) {
        output += computeFixed(1, rest_input_vector);
    }

    return output;
}

fn computeFixed(comptime vector_len: comptime_int, input_vector: []const f32) f32 {
    @setFloatMode(.Optimized);

    var output_vector: @Vector(vector_len, f32) = @splat(0);
    var offset: usize = 0;

    while (offset + vector_len <= input_vector.len) : (offset += vector_len) {
        output_vector += input_vector[offset..][0..vector_len].*;
    }

    return @reduce(.Add, output_vector);
}

test "reduce_add" {
    const allocator = std.testing.allocator;

    for (0..2) |vector_len_subtrahend| {
        const vector_len = constants.max_vector_len - vector_len_subtrahend;
        const input_vector = try allocator.alloc(f32, vector_len);

        defer allocator.free(input_vector);

        var expected_output: f32 = 0;

        for (0..input_vector.len) |index| {
            const value: f32 = @floatFromInt(index + 1);

            input_vector[index] = value;
            expected_output += value;
        }

        try std.testing.expectApproxEqRel(
            expected_output,
            compute(input_vector),
            std.math.floatEps(f32),
        );
    }
}
