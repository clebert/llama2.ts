const std = @import("std");

const constants = @import("constants.zig");

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

    var offset: usize = 0;

    while (offset + vector_len <= input_vector.len) : (offset += vector_len) {
        output_vector[offset..][0..vector_len].* = @exp(
            @as(@Vector(vector_len, f32), input_vector[offset..][0..vector_len].*),
        );
    }
}

test "compute exp" {
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
            const value: f32 = @floatFromInt(index + 1);

            try std.testing.expectApproxEqRel(
                @exp(value),
                output_vector[index],
                std.math.floatEps(f32),
            );
        }
    }
}
