import torch
from torch import nn, optim
import numpy as np
import argparse
import time
from VITModel import save_checkpoint, save_experiment, prepare_data, ViTForClassfication


np.random.seed(0)
torch.manual_seed(0)

batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "patch_size": 8,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 4 * 768,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 64,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": False,
    "device" : device
}

def main():
    model = ViTForClassfication(config).to(device)
    model.load_state_dict(torch.load('weights/eurosat_weights.pth',map_location=torch.device(device)))

    input_names = [ "actual_input" ]
    output_names = [ "output" ]
    dummy_input = torch.randn(32, 3, 64, 64).cuda()
    torch.onnx.export(model,
                 dummy_input,
                 "onnx_models/vit_eurosat_opset14.onnx",
                 verbose=False,
                 input_names=input_names,
                 output_names=output_names,
                 export_params=True,
                 opset_version=14
                 )

if __name__ == '__main__':
    main()
