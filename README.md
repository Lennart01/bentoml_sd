# Simple BentoML StableDiffusion Demo
I strongly recommend running the model download in tmux or screen, as it takes a while to download the models.
## Pip requirements

```bash
pip install -r requirements.txt
```

## Run

Download models and import them into bentoml

```bash
python3 models.py
```

Run the server

```bash
BENTOML_CONFIG=bento_config.yaml bentoml serve <service_name.py>:svc
```