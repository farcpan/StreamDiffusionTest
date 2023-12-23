import cv2
import torch
from PIL import Image
import numpy as np
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

video_cap = cv2.VideoCapture(0)

# You can load any models using diffuser's StableDiffusionPipeline
model_name = "KBlueLeaf/kohaku-v2.1"
#model_name = "xyn-ai/anything-v4.0"
pipe = StableDiffusionPipeline.from_pretrained(model_name).to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

# Wrap the pipeline in StreamDiffusion
stream = StreamDiffusion(
    pipe,
    t_index_list=[32, 45],
    torch_dtype=torch.float16,
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

# image
ret, frame = video_cap.read()
imgCV_RGB = frame[:, :, ::-1]
init_image = Image.fromarray(imgCV_RGB)

# Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(2):
    stream(init_image)

# Run the stream infinitely
while True:
    ret, frame = video_cap.read()
    imgCV_RGB = frame[:, :, ::-1]
    x_output = stream(Image.fromarray(imgCV_RGB))

    output_image = postprocess_image(x_output, output_type="pil")[0]

    new_image = np.array(output_image, dtype=np.uint8)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("output", new_image)
    #input_response = input("Press Enter to continue or type 'stop' to exit: ")
    #if input_response == "stop":
    #   break
    cv2.waitKey(10)
