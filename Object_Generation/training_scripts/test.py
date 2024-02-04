from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from lora_diffusion import tune_lora_scale, patch_pipe

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

prompt = "a blue <s1><s2>"

patch_pipe(
    pipe,
   "exps/examples/vase/lora_weight.safetensors",
    patch_text=True,
    patch_ti=True,
    patch_unet=True,
)

tune_lora_scale(pipe.unet, 0.7)
tune_lora_scale(pipe.text_encoder, 0.7)
for i in range(5):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
    image.save(("exps/examples/vase/fusion_0.7_image_"+str(i)+".png"))

tune_lora_scale(pipe.unet, 1)
tune_lora_scale(pipe.text_encoder, 1)
for i in range(5):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
    image.save(("exps/examples/vase/fusion_1_image_"+str(i)+".png"))

