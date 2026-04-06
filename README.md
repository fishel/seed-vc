# Livonian Seed-VC

TL;DR: adapted copy of https://github.com/plachtaa/seed-vc

To run voice conversion install the requirements:

```bash
pip install -r requirements.txt
```

And run the command:

```bash
inference_v2.py \
   --source input_audio.wav \
   --target target_voice.wav \
   --diffusion-steps 50 \
   --output output_directory_path \
   --ar-checkpoint-path autoregressive_model_checkpoint \
   --cfm-checkpoint-path continuous_flow_matching_model_checkpoint \
   --anonymization-only true
```

Or apply to a whole directory of input files:

```bash
inference_v2.py \
   --source input_directory \
   ...
```
