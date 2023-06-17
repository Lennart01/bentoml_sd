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

input_spec = JSON(pydantic_model=InputSchema)

@svc.api(input=input_spec, output=Image())
def txt2img(input_data: input_spec) -> Image():
    print(input_data)
    prompt = input_data.prompt
    negative_prompt = input_data.negative_prompt
    num_inference_steps = input_data.num_inference_steps
    guidance_scale = input_data.guidance_scale
    output = stable_diffusion_runner.run(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    images = output[0]
    return images[0]
