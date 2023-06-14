import bentoml
from bentoml.io import Image, JSON

bento_model = bentoml.diffusers.get("sdv2_1:latest")
stable_diffusion_runner = bento_model.to_runner()

svc = bentoml.Service("sdv2_1", runners=[stable_diffusion_runner])

@svc.api(input=JSON(), output=Image())
def txt2img(input_data):
    print(input_data)
    output = stable_diffusion_runner.run(**input_data)
    print(output)
    images, _ = output
    return images[0]
