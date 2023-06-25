# import sdv2_1 from huggingface into bentoml

import bentoml

bentoml.diffusers.import_model(
    "sdv2_1", 
    "stabilityai/stable-diffusion-2-1"
)
