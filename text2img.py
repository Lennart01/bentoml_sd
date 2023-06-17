import bentoml
from bentoml.io import Image, JSON
from pydantic import BaseModel

bento_model = bentoml.diffusers.get("sdv2_1:latest")
stable_diffusion_runner = bento_model.to_runner()

svc = bentoml.Service("sdv2_1", runners=[stable_diffusion_runner])

class InputSchema(BaseModel):
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    guidance_scale: float

@svc.api(input=InputSchema, output=Image)
def txt2img(input_data):
    print(input_data)
    output = stable_diffusion_runner.run(**input_data)
    print(output)
    images = output[0]
    return images[0]
