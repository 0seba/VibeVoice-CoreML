#!/usr/bin/env python3
"""
Numpy-only CoreML Inference Demo for VibeVoice Streaming Model.

This script demonstrates the pure numpy inference pipeline for VibeVoice,
eliminating PyTorch dependency while leveraging CoreML models for neural
network computations on Apple Silicon.

Key Benefits:
- No PyTorch dependency for inference
- Pure numpy operations throughout
- Leverages CoreML for Neural Engine acceleration

Usage:
    python demo/numpy_realtime_inference.py \
        --model_path microsoft/VibeVoice-Realtime-0.5B \
        --txt_path demo/text_examples/1p_vibevoice.txt \
        --speaker_name Wayne \
        --output_dir ./outputs
"""

import argparse
import os
import time
from typing import Tuple, Dict, Any, Optional

import numpy as np

# Import CoreML tools
import coremltools as ct

# Import VibeVoice components
# from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoiceane.vibevoice_numpy_streaming_processor import (
    VibeVoiceNumpyStreamingProcessor,
)
from vibevoiceane.configuration_vibevoice_streaming import VibeVoiceStreamingConfig

# Import the numpy-only inference module
from vibevoiceane.wrapped_inference import (
    NumPyStreamingGenerator,
)

MODEL_BASE_PATH = "/Users/seba/Documents/mydeving/VibeVoice-Realtime-0.5B-CoreML"


class VoiceMapper:
    """Maps speaker names to voice file paths and loads them."""

    def __init__(self):
        self.setup_voice_presets()

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(MODEL_BASE_PATH, "voices/streaming_model")

        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        self.voice_presets = {}
        # Scan for .npz files first
        files = [
            f
            for f in os.listdir(voices_dir)
            if (f.lower().endswith(".npz") or f.lower().endswith(".pt"))
            and os.path.isfile(os.path.join(voices_dir, f))
        ]

        for f in files:
            name = os.path.splitext(f)[0]
            if name in self.voice_presets and f.endswith(".pt"):
                continue  # Prefer npz if already found

            full_path = os.path.join(voices_dir, f)
            self.voice_presets[name] = full_path

        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {
            name: path
            for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name."""
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]

        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if (
                preset_name.lower() in speaker_lower
                or speaker_lower in preset_name.lower()
            ):
                return path

        if not self.voice_presets:
            raise ValueError("No voices available")

        default_voice = list(self.voice_presets.values())[0]
        print(
            f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}"
        )
        return default_voice

    def load_voice(self, voice_path: str) -> Dict[str, Any]:
        """Load voice file (prompts) into a dictionary structure."""
        if voice_path.endswith(".pt"):
            print(f"Warning: Loading .pt file {voice_path}. This requires torch.")
            import torch

            return torch.load(voice_path, map_location="cpu", weights_only=False)

        # Load .npz
        print(f"Loading voice from .npz: {voice_path}")
        data = np.load(voice_path)

        # Reconstruct structure
        result = {}

        for key in data.files:
            # Keys like "lm/past_key_values/key" or "neg_tts_lm/last_hidden_state"
            parts = key.split("/")
            current = result
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set value
            current[parts[-1]] = data[key]

        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Numpy-only VibeVoice Streaming Inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-Realtime-0.5B",
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--txt_path",
        type=str,
        # default="demo/text_examples/1p_vibevoice copy.txt",
        default="demo/text_examples/1p_vibevoice.txt",
        help="Path to the txt file containing the script",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="Wayne",
        help="Speaker name for voice cloning",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output audio files",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="CFG scale for generation (default: 1.5)",
    )
    parser.add_argument(
        "--use_numpy",
        action="store_true",
        default=True,
        help="Use numpy-only generation (default: True)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def load_model_configuration(model_path: str) -> VibeVoiceStreamingConfig:
    """
    Load model configuration from the specified path.
    """
    config = VibeVoiceStreamingConfig.from_pretrained(model_path)
    return config


def create_numpy_generator(
    weights: Optional[Dict[str, float]] = None,
    config=None,
) -> NumPyStreamingGenerator:
    """
    Create a numpy-only streaming generator.
    """
    # Create the numpy generator
    numpy_gen = NumPyStreamingGenerator(
        config=config,
        lm_mlmodel_path=os.path.join(
            MODEL_BASE_PATH, "vibe_voice_lm_model_seqlen_32.mlpackage"
        ),
        tts_lm_mlmodel_path=os.path.join(
            MODEL_BASE_PATH, "vibevoice_tts_lm_model_fused_seqlen_8.mlpackage"
        ),
        diffusion_head_mlmodel_path=os.path.join(
            MODEL_BASE_PATH, "diffusion_head_model.mlpackage"
        ),
        acoustic_detokenizer_mlmodel_path=os.path.join(
            MODEL_BASE_PATH, "decoder_coreml_12_ne.mlpackage"
        ),
        speech_connector_mlmodel_path=os.path.join(
            MODEL_BASE_PATH, "acoustic_connector.mlpackage"
        ),
        eos_classifier_mlmodel_path=os.path.join(
            MODEL_BASE_PATH, "tts_eos_classifier.mlpackage"
        ),
        embed_tokens_path=os.path.join(MODEL_BASE_PATH, "vibevoice_embeddings.npy"),
        tts_input_types_path=os.path.join(MODEL_BASE_PATH, "tts_input_types.npy"),
        speech_scaling_factor=weights.get("speech_scaling_factor", 1.0),
        speech_bias_factor=weights.get("speech_bias_factor", 0.0),
        acoustic_vae_dim=config.acoustic_vae_dim if config else 64,
        hidden_size=config.decoder_config.hidden_size if config else 896,
        ddpm_num_inference_steps=5,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    return numpy_gen


def prepare_inputs_numpy(
    processor: VibeVoiceNumpyStreamingProcessor,
    text: str,
    all_prefilled_outputs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare numpy inputs for inference.
    """
    # Process input
    # Note: We need to ensure we don't pass 'pt' tensors if avoiding torch
    # The processor returns transformers BatchEncoding.
    # If return_tensors is not specified, it returns lists usually.
    # We will convert lists to numpy.

    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors=None,  # Return lists
        return_attention_mask=True,
    )

    # Convert lists to numpy
    tts_text_ids = np.array(inputs["tts_text_ids"], dtype=np.int64)
    tts_lm_input_ids = np.array(inputs["tts_lm_input_ids"], dtype=np.int64)

    # Ensure 2D (batch, seq)
    if tts_text_ids.ndim == 1:
        tts_text_ids = tts_text_ids[np.newaxis, :]
    if tts_lm_input_ids.ndim == 1:
        tts_lm_input_ids = tts_lm_input_ids[np.newaxis, :]

    return tts_text_ids, tts_lm_input_ids


def save_audio_numpy(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 24000,
):
    """
    Save numpy audio array to WAV file.
    """
    import wave

    # Ensure audio is in the correct format
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Save as WAV
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    print(f"Saved audio to: {output_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("VibeVoice Numpy-Only CoreML Inference Demo")
    print("=" * 60)

    # Validate txt file
    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        return

    # Read script
    with open(args.txt_path, "r", encoding="utf-8") as f:
        scripts = f.read().strip()

    if not scripts:
        print("Error: No valid scripts found in the txt file")
        return

    full_script = scripts.replace("'", "'").replace('"', '"').replace('"', '"')

    print(f"\nLoading processor from {args.model_path}")
    processor = VibeVoiceNumpyStreamingProcessor.from_pretrained(args.model_path)

    # Load voice
    voice_mapper = VoiceMapper()
    voice_sample_path = voice_mapper.get_voice_path(args.speaker_name)
    all_prefilled_outputs = voice_mapper.load_voice(voice_sample_path)

    print(f"Using voice: {voice_sample_path}")

    # Config
    config = load_model_configuration(args.model_path)

    # Weights - Hardcoded for now as we removed torch extraction
    weights = {
        "speech_bias_factor": -0.0703125,
        "speech_scaling_factor": 0.2333984375,
    }

    # Create numpy generator
    print("\nCreating numpy-only generator...")
    numpy_gen = create_numpy_generator(weights=weights, config=config)

    # Prepare inputs
    print("\nPreparing inputs...")
    tts_text_ids, tts_lm_input_ids = prepare_inputs_numpy(
        processor, full_script, all_prefilled_outputs
    )
    print(f"Text tokens shape: {tts_text_ids.shape}")
    print(f"TTS LM input shape: {tts_lm_input_ids.shape}")

    # Generate
    print(f"\nStarting numpy generation with cfg_scale={args.cfg_scale}...")
    start_time = time.time()

    outputs = numpy_gen.generate(
        tts_text_ids=tts_text_ids,
        tts_lm_input_ids=tts_lm_input_ids,
        all_prefilled_outputs=all_prefilled_outputs,
        cfg_scale=args.cfg_scale,
        max_length=config.decoder_config.max_position_embeddings,
        return_speech=True,
        verbose=args.verbose,
    )

    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")

    # Process output
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        sample_rate = 24000
        audio_samples = (
            outputs.speech_outputs[0].shape[-1]
            if len(outputs.speech_outputs[0].shape) > 0
            else len(outputs.speech_outputs[0])
        )
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

        print(f"\nGenerated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")

        # Save output
        os.makedirs(args.output_dir, exist_ok=True)
        txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
        output_path = os.path.join(
            args.output_dir, f"{txt_filename}_{args.speaker_name}_numpy_generated.wav"
        )
        save_audio_numpy(outputs.speech_outputs[0], output_path, sample_rate)

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY (Numpy-Only)")
    print("=" * 60)
    print(f"Input file: {args.txt_path}")
    print(f"Output file: {output_path}")
    print(f"Speaker: {args.speaker_name}")
    print(f"Generation time: {generation_time:.2f} seconds")
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        print(f"Audio duration: {audio_duration:.2f} seconds")
        print(f"RTF: {rtf:.2f}x")
    print(f"Reach max step sample: {outputs.reach_max_step_sample}")
    print("=" * 60)

    print("\nNumpy-only generation completed successfully!")


if __name__ == "__main__":
    main()
