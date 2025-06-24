<!---
Copyright 2022 - The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# **ControlNet Customed with 🤗 Diffusers**

This repository provides examples and instructions for fine-tuning ControlNet on the circle-filling dataset using 🤗 Diffusers. You can train with PyTorch or Flax/JAX, leverage Accelerate for distributed/multi-GPU training, and apply various memory optimizations for GPUs of different sizes.

---

## 📦 Installation

1. **Clone and install Diffusers**

   ```bash
   pip install -e .
   ```
2. **Install example requirements**

   ```bash
   cd examples/controlnet
   pip install -r requirements.txt
   ```
3. **Initialize Accelerate**

   * Interactive setup:

     ```bash
     accelerate config
     ```
   * Default configuration:

     ```bash
     accelerate config default
     ```
   * Non-interactive (e.g., notebook):

     ```python
     from accelerate.utils import write_basic_config
     write_basic_config()
     ```

---

## 📂 Dataset

We use the circle-filling dataset (re-hosted for 🤗 Datasets compatibility) under `fusing/fill50k`. Dataloading is handled inside the training script.

## 🖼️ Download Conditioning Images

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

---

## 🚀 Training (PyTorch)

```bash
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="output"

accelerate launch my_train_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=fusing/fill50k \
  --resolution=512 \
  --learning_rate=1e-5 \
  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
  --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
  --train_batch_size=4
  --report_to=wandb
```


## 📖 References

* [Diffusers Documentation](https://github.com/huggingface/diffusers)
* [ControlNet Paper](https://arxiv.org/abs/2302.05543)
* [Accelerate Documentation](https://huggingface.co/docs/accelerate)
* [DeepSpeed Zero3](https://www.deepspeed.ai/tutorials/zero/)
