const std = @import("std");

const constants = @import("constants.zig");

pub fn compute(input_vector_1: []const f32, input_vector_2: []const f32) f32 {
    std.debug.assert(input_vector_1.len == input_vector_2.len);

    comptime var vector_len = constants.max_vector_len;

    var output: f32 = 0;
    var rest_input_vector_1 = input_vector_1;
    var rest_input_vector_2 = input_vector_2;

    inline while (vector_len >= constants.min_vector_len) : (vector_len /= 2) {
        if (rest_input_vector_1.len >= vector_len) {
            output += computeFixed(vector_len, rest_input_vector_1, rest_input_vector_2);

            const rest_len = rest_input_vector_1.len % vector_len;

            rest_input_vector_1 = input_vector_1[input_vector_1.len - rest_len ..];
            rest_input_vector_2 = input_vector_2[input_vector_2.len - rest_len ..];
        }
    }

    if (rest_input_vector_1.len > 0) {
        output += computeFixed(1, rest_input_vector_1, rest_input_vector_2);
    }

    return output;
}

fn computeFixed(
    comptime vector_len: comptime_int,
    input_vector_1: []const f32,
    input_vector_2: []const f32,
) f32 {
    @setFloatMode(.Optimized);

    var output_vector: @Vector(vector_len, f32) = @splat(0);
    var offset: usize = 0;

    while (offset + vector_len <= input_vector_1.len) : (offset += vector_len) {
        output_vector +=
            @as(@Vector(vector_len, f32), input_vector_1[offset..][0..vector_len].*) *
            @as(@Vector(vector_len, f32), input_vector_2[offset..][0..vector_len].*);
    }

    return @reduce(.Add, output_vector);
}

test "compute dot product" {
    const allocator = std.testing.allocator;

    for (0..2) |vector_len_subtrahend| {
        const vector_len = constants.max_vector_len - vector_len_subtrahend;
        const input_vector_1 = try allocator.alloc(f32, vector_len);
        const input_vector_2 = try allocator.alloc(f32, vector_len);

        defer allocator.free(input_vector_1);
        defer allocator.free(input_vector_2);

        var expected_output: f32 = 0;

        for (0..input_vector_1.len) |index| {
            const value: f32 = @floatFromInt(index + 1);

            input_vector_1[index] = value;
            input_vector_2[index] = value;
            expected_output += value * value;
        }

        try std.testing.expectApproxEqRel(
            expected_output,
            compute(input_vector_1, input_vector_2),
            std.math.floatEps(f32),
        );
    }
}
