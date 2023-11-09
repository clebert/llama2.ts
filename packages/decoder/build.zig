const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});

    buildWasmLib(b, "addition_layer", optimize);
    buildWasmLib(b, "attention_layer", optimize);
    buildWasmLib(b, "fnn_layer", optimize);
    buildWasmLib(b, "linear_layer", optimize);
    buildUnitTests(b);
}

const wasm_target = std.zig.CrossTarget{
    .cpu_arch = .wasm32,
    .os_tag = .freestanding,
    .cpu_features_add = std.Target.Cpu.Feature.feature_set_fns(std.Target.wasm.Feature).featureSet(
        &[_]std.Target.wasm.Feature{.simd128},
    ),
};

fn buildWasmLib(b: *std.Build, comptime name: []const u8, optimize: std.builtin.OptimizeMode) void {
    const lib = b.addSharedLibrary(.{
        .name = name,
        .root_source_file = .{ .path = "src/" ++ name ++ ".zig" },
        .target = wasm_target,
        .optimize = optimize,
    });

    lib.rdynamic = true;

    b.installArtifact(lib);
}

fn buildUnitTests(b: *std.Build) void {
    const test_step = b.step("test", "Run unit tests");

    const simd_test = b.addTest(.{
        .root_source_file = .{ .path = "src/simd/test.zig" },
        .optimize = std.builtin.OptimizeMode.Debug,
    });

    const run_tests = b.addRunArtifact(simd_test);

    test_step.dependOn(&run_tests.step);
}
