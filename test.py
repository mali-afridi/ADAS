# import torch
# import torchvision
# import onnx
# import onnxruntime

# # Load your PyTorch model
# # model = torchvision.models.resnet18(pretrained=True)
# # model.eval()

# model = dict(test_cfg=dict(conf_threshold=0.43))

# # Define example input tensor
# example_input = torch.randn(1, 3, 224, 224)  # Adjust the shape as per your model's input requirements

# # Export the model to ONNX format
# onnx_file_path = "clrernet.onnx"
# torch.onnx.export(model, example_input, onnx_file_path, export_params=True, opset_version=11, input_names=['input'], output_names=['output'])

# # Load the ONNX model
# onnx_model = onnx.load(onnx_file_path)

# # Create an ONNX Runtime Inference Session
# ort_session = onnxruntime.InferenceSession(onnx_file_path)

# # Run inference with ONNX Runtime
# ort_inputs = {ort_session.get_inputs()[0].name: example_input.numpy()}
# ort_outs = ort_session.run(None, ort_inputs)

# # Process the output as per your requirements
# print("Output shape:", ort_outs[0].shape)

import numpy as np
import matplotlib.pyplot as plt

training_data = np.load('/home/sami/Desktop/Code/CLRerNet/dataset2/culane/list/train_diffs.npz', allow_pickle=True, mmap_mode='r')
training_data =  training_data['data']

plt.imshow(training_data)
plt.show()