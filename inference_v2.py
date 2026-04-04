import os
import argparse
import torch
import yaml
import soundfile as sf
import time

from accelerate import Accelerator
from modules.commons import str2bool
from os.path import isdir, isfile

# Set up device and torch configurations
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

# Global variables to store model instances
vc_wrapper_v2 = None


def load_v2_models(args):
    """Load V2 models using the wrapper from app.py"""
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                cfm_checkpoint_path=args.cfm_checkpoint_path)
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            # Experimental feature to reduce compilation times, will be on by default in future
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()
        # vc_wrapper.compile_cfm()

    return vc_wrapper


def convert_voice_v2(source_audio_path, target_audio_path, args):
    """Convert voice using V2 model"""
    global vc_wrapper_v2
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args)

    # Use the generator function but collect all outputs
    generator = vc_wrapper_v2.convert_voice_with_streaming(
        source_audio_path=source_audio_path,
        target_audio_path=target_audio_path,
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        intelligebility_cfg_rate=args.intelligibility_cfg_rate,
        similarity_cfg_rate=args.similarity_cfg_rate,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        convert_style=args.convert_style,
        anonymization_only=args.anonymization_only,
        device=device,
        dtype=dtype,
        stream_output=True
    )

    # Collect all outputs from the generator
    for output in generator:
        _, full_audio = output
    result = full_audio

    if result is None:
        raise Exception("Error: Failed to convert voice")
    return result


def get_output_path(src, tgt, dir):
    filename = f"{src}_vcv2_{tgt}.wav"
    output_path = os.path.join(dir, filename)
    return output_path


def save_it(converted_audio, src_file, tgt_file, output_dir):
    # Save the converted audio
    source_name = os.path.basename(src_file).split(".")[0]
    target_name = os.path.basename(tgt_file).split(".")[0]

    # Create a descriptive filename
    output_path = get_output_path(source_name, target_name, output_dir)

    save_sr, converted_audio = converted_audio
    sf.write(output_path, converted_audio, save_sr)


def convert_and_save_file(src_path, tgt_path, params, proc_idx=None):
    print(f"REPORT Converting {src_path} to {tgt_path}" + ("" if proc_idx is None else f", proc {proc_idx}"))
    converted_audio = convert_voice_v2(src_path, tgt_path, params)
    save_it(converted_audio, src_path, tgt_path, params.output)


def convert_and_save_dir(dir_path, tgt_path, params, acc=None):
    paths = sorted([os.path.join(dir_path, src_file)
                    for src_file in os.listdir(dir_path)
                    if isfile(os.path.join(dir_path, src_file))])
    count = 0

    for i, path in enumerate(paths):
        if acc is not None and i % acc.num_processes == acc.local_process_index:
            output_path = get_output_path(path, tgt_path, params.output)

            if os.path.exists(output_path):
                print(f"REPORT Skipping {path}, output {output_path} already exists")
            else:
                count += 1
                try:
                    convert_and_save_file(path, tgt_path, params, proc_idx=acc.local_process_index)
                except:
                    print(f"REPORT Caught an exception on {path}, skipping")

    return len(paths)


def main(args):
    acc = Accelerator()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    start_time = time.time()

    if isdir(args.source):
        num = convert_and_save_dir(args.source, args.target, args, acc)
    else:
        convert_and_save_file(args.source, args.target, args)
        num = 1
    end_time = time.time()

    len = end_time - start_time

    print(f"REPORT Converted all {num} files in {len} time; average time per file: {len / num}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Conversion Inference Script")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to source audio file")
    parser.add_argument("--target", type=str, required=True,
                        help="Path to target/reference audio file")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory for converted audio")
    parser.add_argument("--diffusion-steps", type=int, default=30,
                        help="Number of diffusion steps")
    parser.add_argument("--length-adjust", type=float, default=1.0,
                        help="Length adjustment factor (<1.0 for speed-up, >1.0 for slow-down)")
    parser.add_argument("--compile", type=bool, default=False,
                        help="Whether to compile the model for faster inference")

    # V2 specific arguments
    parser.add_argument("--intelligibility-cfg-rate", type=float, default=0.7,
                        help="Intelligibility CFG rate for V2 model")
    parser.add_argument("--similarity-cfg-rate", type=float, default=0.7,
                        help="Similarity CFG rate for V2 model")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter for V2 model")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature sampling parameter for V2 model")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty for V2 model")
    parser.add_argument("--convert-style", type=str2bool, default=False,
                        help="Convert style/emotion/accent for V2 model")
    parser.add_argument("--anonymization-only", type=str2bool, default=False,
                        help="Anonymization only mode for V2 model")

    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")

    args = parser.parse_args()
    main(args)
