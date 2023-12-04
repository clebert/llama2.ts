const std = @import("std");

pub const max_vector_len = @max(std.atomic.cache_line / @sizeOf(f32), 32);
pub const min_vector_len = std.simd.suggestVectorSize(f32) orelse 4;

comptime {
    if (max_vector_len % min_vector_len != 0) {
        @panic("invalid min/max vector length");
    }
}
