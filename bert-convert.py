
from pathlib import Path
from transformers.convert_graph_to_onnx import convert

convert(framework="pt", model="bert-base-cased", output=Path("onnx/bert-base-cased.onnx"), opset=11)
