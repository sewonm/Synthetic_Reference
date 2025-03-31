import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image, make_image_grid
from google.colab import drive
from PIL import Image

drive.mount('/content/drive')

image_path = "/content/drive/MyDrive/pose1.webp"
image = Image.open(image_path).convert("RGB")
image.show()

def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map

depth_estimator = pipeline("depth-estimation")
depth_map = get_depth_map(image, depth_estimator).unsqueeze(0).half().to("cuda")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, use_safetensors=True
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "a photorealistic female model in an action pose, dramatic studio lighting, wearing a dress, standing in a clean studio environment, high detail, full body"
output = pipe(prompt, image=image, control_image=depth_map).images[0]

make_image_grid([image, output], rows=1, cols=2)
