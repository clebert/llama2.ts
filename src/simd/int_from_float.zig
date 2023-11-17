const std = @import("std");

const constants = @import("constants.zig");

pub fn compute(comptime T: type, input_vector: []const f32, output_vector: []T) void {
    std.debug.assert(input_vector.len == output_vector.len);

    comptime var vector_len = constants.max_vector_len;

    var rest_input_vector = input_vector;
    var rest_output_vector = output_vector;

    inline while (vector_len >= constants.min_vector_len) : (vector_len /= 2) {
        if (rest_input_vector.len >= vector_len) {
            computeFixed(T, vector_len, rest_input_vector, rest_output_vector);

            const rest_len = rest_input_vector.len % vector_len;

            rest_input_vector = input_vector[input_vector.len - rest_len ..];
            rest_output_vector = output_vector[output_vector.len - rest_len ..];
        }
    }

    if (rest_input_vector.len > 0) {
        computeFixed(T, 1, rest_input_vector, rest_output_vector);
    }
}

fn computeFixed(
    comptime T: type,
    comptime vector_len: comptime_int,
    input_vector: []const f32,
    output_vector: []T,
) void {
    @setFloatMode(.Optimized);

    var offset: usize = 0;

    while (offset + vector_len <= input_vector.len) : (offset += vector_len) {
        output_vector[offset..][0..vector_len].* = @as(
            @Vector(vector_len, T),
            @intFromFloat(@as(@Vector(vector_len, f32), input_vector[offset..][0..vector_len].*)),
        );
    }
}

test "int_from_float" {
    const allocator = std.testing.allocator;

    for (0..2) |vector_len_subtrahend| {
        const vector_len = constants.max_vector_len - vector_len_subtrahend;
        const input_vector = try allocator.alloc(f32, vector_len);
        const output_vector = try allocator.alloc(i8, vector_len);

        defer allocator.free(input_vector);
        defer allocator.free(output_vector);

        for (0..input_vector.len) |index| {
            input_vector[index] = @as(f32, @floatFromInt(index + 1)) + 0.9;
        }

        compute(i8, input_vector, output_vector);

        for (0..output_vector.len) |index| {
            try std.testing.expectEqual(@as(i8, @intCast(index + 1)), output_vector[index]);
        }
    }
}
