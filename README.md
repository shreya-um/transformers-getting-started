# Transformers getting started
Basics: [Transformer Neural Networks: A Step-by-Step Breakdown](https://builtin.com/artificial-intelligence/transformer-neural-network)

## Install all python dependencies (tested on python 3.11)
  1. clone the repo, go the folder
  2. Create a python venv `python3.11 -m venv .transformersEnv`.
  3. source the venv `source .transformersEnv/bin/activate`.
  4. install dependencies ` pip install -r requirements.txt`.
  5. run convert script `python convert.py`
This script creates a onnx folder with Bert onnx model with precision (float32). \
Use [Netron](https://netron.app/) to verify and visualise onnx model

## onnx to MLIR (LLVM dialect)
Use [onnx-mlir](https://github.com/onnx/onnx-mlir) to convert onnx models to LLVM dialect.
1. [Build on local](https://github.com/onnx/onnx-mlir/blob/main/docs/BuildONNX.md) or for quick conversions use Docker img [Instructions](https://github.com/onnx/onnx-mlir/blob/main/docs/Docker.md)
2. Follow [Environment Variables Setup](https://github.com/onnx/onnx-mlir/blob/main/docs/mnist_example/README.md#environment-variables-setup)
3. To generate LLVM Dialect`onnx-mlir -O3 --EmitLLVMIR <onnx model file>`
4. (optional) [creating executable binary example](https://github.com/onnx/onnx-mlir/blob/main/docs/mnist_example/README.md)

## LLVM Dialect to LLVM IR
1. Build MLIR on local (preffered debug version) [MLIR getting started](https://mlir.llvm.org/getting_started/)
2. (Optional) set path variable to mlir project bin folder
3. use `mlir-translate <input-file-LLVM-dialect> --mlir-to-llvmir -o <output.ll>` to get LLVM IR.
