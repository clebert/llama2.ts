export function computeArgmax(inputVector: Float32Array): number {
  let maxIndex = 0;
  let maxInput = inputVector[maxIndex]!;

  for (let index = 1; index < inputVector.length; index += 1) {
    const input = inputVector[index]!;

    if (input > maxInput) {
      maxIndex = index;
      maxInput = input;
    }
  }

  return maxIndex;
}
