const std = @import("std");

const constants = @import("constants.zig");

pub fn compute(
    input_vector_1: []const f32,
    input_vector_2: []const f32,
    output_vector: []f32,
) void {
    std.debug.assert(input_vector_1.len == input_vector_2.len);
    std.debug.assert(input_vector_1.len == output_vector.len);

    comptime var vector_len = constants.max_vector_len;

    var rest_input_vector_1 = input_vector_1;
    var rest_input_vector_2 = input_vector_2;
    var rest_output_vector = output_vector;

    inline while (vector_len >= constants.min_vector_len) : (vector_len /= 2) {
        if (rest_input_vector_1.len >= vector_len) {
            computeFixed(vector_len, rest_input_vector_1, rest_input_vector_2, rest_output_vector);

            const rest_len = rest_input_vector_1.len % vector_len;

            rest_input_vector_1 = input_vector_1[input_vector_1.len - rest_len ..];
            rest_input_vector_2 = input_vector_2[input_vector_2.len - rest_len ..];
            rest_output_vector = output_vector[output_vector.len - rest_len ..];
        }
    }

    if (rest_input_vector_1.len > 0) {
        computeFixed(1, rest_input_vector_1, rest_input_vector_2, rest_output_vector);
    }
}

fn computeFixed(
    comptime vector_len: comptime_int,
    input_vector_1: []const f32,
    input_vector_2: []const f32,
    output_vector: []f32,
) void {
    @setFloatMode(.Optimized);

    var offset: usize = 0;

    while (offset + vector_len <= input_vector_1.len) : (offset += vector_len) {
        output_vector[offset..][0..vector_len].* =
            @as(@Vector(vector_len, f32), input_vector_1[offset..][0..vector_len].*) *
            @as(@Vector(vector_len, f32), input_vector_2[offset..][0..vector_len].*);
    }
}

test "hadamard_product" {
    const allocator = std.testing.allocator;

    for (0..2) |vector_len_subtrahend| {
        const vector_len = constants.max_vector_len - vector_len_subtrahend;
        const input_vector_1 = try allocator.alloc(f32, vector_len);
        const input_vector_2 = try allocator.alloc(f32, vector_len);
        const output_vector = try allocator.alloc(f32, vector_len);

        defer allocator.free(input_vector_1);
        defer allocator.free(input_vector_2);
        defer allocator.free(output_vector);

        for (0..input_vector_1.len) |index| {
            const value: f32 = @floatFromInt(index + 1);

            input_vector_1[index] = value;
            input_vector_2[index] = value;
        }

        compute(input_vector_1, input_vector_2, output_vector);

        for (0..output_vector.len) |index| {
            const value: f32 = @floatFromInt(index + 1);

            try std.testing.expectApproxEqRel(
                value * value,
                output_vector[index],
                std.math.floatEps(f32),
            );
        }
    }
}
