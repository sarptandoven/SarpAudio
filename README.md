```bash
pip install -r requirements.txt
```

```bash
pip install -r requirements-reuse.txt
```

```bash
python app.py
```

```bash
python src/inference.py \
  --voice-sample reference.wav \
  --prompt 'A woman speaks warmly, "Hello, how are you today?"' \
  --output output.wav \
  --cfg-scale 2.5 \
  --stg-scale 1.5
```

```python
from src.inference_server import TTSServer

server = TTSServer(device="cuda")
server.generate_to_file(
    prompt='A woman speaks warmly, "Hello, how are you today?"',
    output="output.wav",
    voice_ref="reference.wav",
)
```

```bash
python generate_5min_demo.py
```

```bash
python src/preprocess.py \
  --dataset-type manifest \
  --index your_data.jsonl \
  --audio-dir /path/to/wavs \
  --output-dir /path/to/preprocessed \
  --checkpoint /path/to/dramabox-audio-components.safetensors \
  --gemma-root /path/to/gemma-3-12b-it-bnb-4bit \
  --max-duration 20.0 \
  --min-duration 2.0
```

```bash
accelerate launch src/train.py --config configs/training_args.example.yaml
```

```bash
python src/validate.py \
  --config configs/val_config.example.yaml \
  --checkpoint /path/to/dramabox-dit-v1.safetensors \
  --full-checkpoint /path/to/dramabox-audio-components.safetensors \
  --gemma-root /path/to/gemma-3-12b-it-bnb-4bit
```
