import onnx
import onnxruntime as ort
import numpy as np
import torch
from torchvision.models import resnet50

# Step 1: Load the ONNX Model
onnx_model_path = 'path_to_your_model.onnx'
onnx_model = onnx.load(onnx_model_path)

# Step 2: Extract the Weights
def extract_onnx_weights(onnx_model):
    session = ort.InferenceSession(onnx_model.SerializeToString())
    weights = {}
    for initializer in onnx_model.graph.initializer:
        name = initializer.name
        data = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
        weights[name] = data
    return weights

weights = extract_onnx_weights(onnx_model)

# Step 3: Define the PyTorch Model
pytorch_model = resnet50(pretrained=False)

# Step 4: Load the Weights into the PyTorch Model
def load_weights_into_pytorch_model(pytorch_model, weights):
    pytorch_dict = pytorch_model.state_dict()

    for name, param in pytorch_dict.items():
        onnx_name = name.replace('.', '_')
        if onnx_name in weights:
            pytorch_dict[name] = torch.from_numpy(weights[onnx_name])

    pytorch_model.load_state_dict(pytorch_dict)

load_weights_into_pytorch_model(pytorch_model, weights)

# Now the PyTorch model should be ready to use
