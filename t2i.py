from PIL import Image
import time
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

# You can load any models using diffuser's StableDiffusionPipeline
#model_name = "stabilityai/sd-turbo"
#model_name = "KBlueLeaf/kohaku-v2.1"
model_name = "xyn-ai/anything-v4.0"
pipe = StableDiffusionPipeline.from_pretrained(model_name).to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

# Wrap the pipeline in StreamDiffusion
# Requires more long steps (len(t_index_list)) in text2image
# You recommend to use cfg_type="none" when text2image
stream = StreamDiffusion(
    pipe,
    t_index_list=[0, 16, 32, 45],
    torch_dtype=torch.float16,
    cfg_type="none",
)

# If the loaded model is not LCM, merge LCM
stream.load_lcm_lora()
stream.fuse_lora()
# Use Tiny VAE for further acceleration
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
# Enable acceleration
pipe.enable_xformers_memory_efficient_attention()

prompt = "muscular boy with shiba inu,night view,window,bedroom"
# Prepare the stream
stream.prepare(prompt)

# Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(4):
    stream()

# Run the stream infinitely
pillow_images = []
for index in range(10):
    start = time.time()
    x_output = stream.txt2img()
    images = postprocess_image(x_output, output_type="pil")
    pillow_images += images
    print(f"Elapsed time: {time.time() - start} [sec]")
    
dst = Image.new('RGB', (512 * len(pillow_images), 512))
for index in range(len(pillow_images)):
    dst.paste(pillow_images[index], (512 * index, 0))
dst.show()


"""
while True:
    x_output = stream.txt2img()
    postprocess_image(x_output, output_type="pil")[0].show()
    input_response = input("Press Enter to continue or type 'stop' to exit: ")
    if input_response == "stop":
        break
"""