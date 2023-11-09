const std = @import("std");

const constants = @import("constants.zig");

// Swish activation function: https://arxiv.org/abs/1710.05941
pub fn compute(input_vector: []const f32, output_vector: []f32) void {
    std.debug.assert(input_vector.len == output_vector.len);

    comptime var vector_len = constants.max_vector_len;

    var rest_input_vector = input_vector;
    var rest_output_vector = output_vector;

    inline while (vector_len >= constants.min_vector_len) : (vector_len /= 2) {
        if (rest_input_vector.len >= vector_len) {
            computeFixed(vector_len, rest_input_vector, rest_output_vector);

            const rest_len = rest_input_vector.len % vector_len;

            rest_input_vector = input_vector[input_vector.len - rest_len ..];
            rest_output_vector = output_vector[output_vector.len - rest_len ..];
        }
    }

    if (rest_input_vector.len > 0) {
        computeFixed(1, rest_input_vector, rest_output_vector);
    }
}

fn computeFixed(
    comptime vector_len: comptime_int,
    input_vector: []const f32,
    output_vector: []f32,
) void {
    @setFloatMode(.Optimized);

    const one_vector: @Vector(vector_len, f32) = @splat(1);

    var offset: usize = 0;

    while (offset + vector_len <= input_vector.len) : (offset += vector_len) {
        const vector: @Vector(vector_len, f32) = input_vector[offset..][0..vector_len].*;
        const sigmoid_vector = one_vector / (one_vector + @exp(-vector));

        output_vector[offset..][0..vector_len].* = vector * sigmoid_vector;
    }
}

test "compute swish" {
    const allocator = std.testing.allocator;

    for (0..2) |vector_len_subtrahend| {
        const vector_len = constants.max_vector_len - vector_len_subtrahend;
        const input_vector = try allocator.alloc(f32, vector_len);
        const output_vector = try allocator.alloc(f32, vector_len);

        defer allocator.free(input_vector);
        defer allocator.free(output_vector);

        for (0..input_vector.len) |index| {
            input_vector[index] = @floatFromInt(index + 1);
        }

        compute(input_vector, output_vector);

        for (0..output_vector.len) |index| {
            try std.testing.expectApproxEqRel(
                computeSwish(@floatFromInt(index + 1)),
                output_vector[index],
                std.math.floatEps(f32),
            );
        }
    }
}

fn computeSwish(x: f32) f32 {
    return x * computeSigmoid(x);
}

fn computeSigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}
