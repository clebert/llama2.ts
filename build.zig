const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});

    buildWasmLib(b, "decoder", "addition_layer", optimize);
    buildWasmLib(b, "decoder", "attention_layer", optimize);
    buildWasmLib(b, "decoder", "fnn_layer", optimize);
    buildWasmLib(b, "decoder", "linear_layer", optimize);
    buildTests(b);
}

const wasm_target = std.zig.CrossTarget{
    .cpu_arch = .wasm32,
    .os_tag = .freestanding,
    .cpu_features_add = std.Target.Cpu.Feature.feature_set_fns(std.Target.wasm.Feature).featureSet(
        &[_]std.Target.wasm.Feature{.simd128},
    ),
};

fn buildWasmLib(
    b: *std.Build,
    comptime package_name: []const u8,
    comptime lib_name: []const u8,
    optimize: std.builtin.OptimizeMode,
) void {
    const wasm_lib = b.addSharedLibrary(.{
        .name = lib_name,
        .root_source_file = .{ .path = "src/" ++ lib_name ++ ".zig" },
        .target = wasm_target,
        .optimize = optimize,
    });

    wasm_lib.rdynamic = true;

    const install_wasm_lib = b.addInstallArtifact(wasm_lib, .{
        .dest_dir = .{ .override = .{ .custom = "../packages/" ++ package_name ++ "/lib/wasm" } },
    });

    b.getInstallStep().dependOn(&install_wasm_lib.step);
}

fn buildTests(b: *std.Build) void {
    const simd_test = b.addTest(.{
        .root_source_file = .{ .path = "src/simd/test.zig" },
        .optimize = std.builtin.OptimizeMode.Debug,
    });

    const run_simd_test = b.addRunArtifact(simd_test);

    b.step("test", "Run tests").dependOn(&run_simd_test.step);
}
