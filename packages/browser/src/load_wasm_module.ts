export async function loadWasmModule(moduleName: string): Promise<WebAssembly.Module> {
  const response = await fetch(`/static/wasm/${moduleName}.wasm`);
  const blob = await response.blob();

  return WebAssembly.compile(await blob.arrayBuffer());
}
