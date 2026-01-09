import os
import io
import re
import time
import threading
import traceback
import queue
import numpy as np
import sounddevice as sd
import uvicorn
import pathlib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    StreamingResponse,
    Response,
    HTMLResponse,
)
from fastapi.middleware.cors import CORSMiddleware
import soundfile
import soxr
import coremltools as ct
from pydantic import BaseModel
from typing import Optional, List
import aiofiles
from huggingface_hub import snapshot_download
import asyncio
from dataclasses import dataclass
import ftfy

try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    PYDUB_AVAILABLE = False

from .vibevoice_numpy_streaming_processor import VibeVoiceNumpyStreamingProcessor
from .configuration_vibevoice_streaming import VibeVoiceStreamingConfig
from .wrapped_inference import NumPyStreamingGenerator

# ============================================================================
# CONFIGURATION & MODEL LOADING
# ============================================================================

REPO_ID = "seba/VibeVoice-Realtime-0.5B-CoreML"
MODEL_PATH_PREFIX = ""
VOICES_DIR = ""


# Global model wrapper
class VibeVoiceModelWrapper:
    def __init__(self):
        self.processor: Optional[VibeVoiceNumpyStreamingProcessor] = None
        self.generator: Optional[NumPyStreamingGenerator] = None
        self.config: Optional[VibeVoiceStreamingConfig] = None
        self.voice_presets = {}
        self.available_voices = {}

    def load(self):
        global MODEL_PATH_PREFIX, VOICES_DIR
        print(
            f"🚀 Downloading/loading model files from Hugging Face Hub repo: {REPO_ID}"
        )
        MODEL_PATH_PREFIX = snapshot_download(repo_id=REPO_ID)
        VOICES_DIR = os.path.join(MODEL_PATH_PREFIX, "voices/streaming_model")

        print(f"Model path: {MODEL_PATH_PREFIX}")

        # Load Processor
        self.processor = VibeVoiceNumpyStreamingProcessor.from_pretrained(
            MODEL_PATH_PREFIX
        )

        # Load Config
        self.config = VibeVoiceStreamingConfig.from_pretrained(MODEL_PATH_PREFIX)

        # Config Weights (Hardcoded from numpy_realtime_inference.py)
        weights = {
            "speech_bias_factor": -0.0703125,
            "speech_scaling_factor": 0.2333984375,
        }

        # Create Generator
        print("Loading CoreML models and creating generator...")
        self.generator = NumPyStreamingGenerator(
            config=self.config,
            lm_mlmodel_path=os.path.join(
                MODEL_PATH_PREFIX, "vibe_voice_lm_model_seqlen_32.mlpackage"
            ),
            tts_lm_mlmodel_path=os.path.join(
                MODEL_PATH_PREFIX, "vibevoice_tts_lm_model_fused_seqlen_8.mlpackage"
            ),
            diffusion_head_mlmodel_path=os.path.join(
                MODEL_PATH_PREFIX, "diffusion_head_model.mlpackage"
            ),
            acoustic_detokenizer_mlmodel_path=os.path.join(
                MODEL_PATH_PREFIX, "decoder_coreml_12_ne.mlpackage"
            ),
            speech_connector_mlmodel_path=os.path.join(
                MODEL_PATH_PREFIX, "acoustic_connector.mlpackage"
            ),
            eos_classifier_mlmodel_path=os.path.join(
                MODEL_PATH_PREFIX, "tts_eos_classifier.mlpackage"
            ),
            embed_tokens_path=os.path.join(
                MODEL_PATH_PREFIX, "vibevoice_embeddings.npy"
            ),
            tts_input_types_path=os.path.join(MODEL_PATH_PREFIX, "tts_input_types.npy"),
            speech_scaling_factor=weights.get("speech_scaling_factor", 1.0),
            speech_bias_factor=weights.get("speech_bias_factor", 0.0),
            acoustic_vae_dim=self.config.acoustic_vae_dim if self.config else 64,
            hidden_size=self.config.decoder_config.hidden_size if self.config else 896,
            ddpm_num_inference_steps=5,  # Default
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )

        self.scan_voices()
        print("✅ Models loaded successfully.")

    def scan_voices(self):
        self.voice_presets = {}
        if not os.path.exists(VOICES_DIR):
            print(f"Warning: Voices directory not found at {VOICES_DIR}")
            return

        files = [
            f
            for f in os.listdir(VOICES_DIR)
            if (f.lower().endswith(".npz") or f.lower().endswith(".pt"))
            and os.path.isfile(os.path.join(VOICES_DIR, f))
        ]

        for f in files:
            name = os.path.splitext(f)[0]
            if name in self.voice_presets and f.endswith(".pt"):
                continue

            full_path = os.path.join(VOICES_DIR, f)
            self.voice_presets[name] = full_path

        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = list(self.voice_presets.keys())
        print(f"Found {len(self.available_voices)} voices.")

    def get_voice_path(self, voice_name: str) -> str:
        if voice_name in self.voice_presets:
            return self.voice_presets[voice_name]

        # Fuzzy match
        voice_lower = voice_name.lower()
        for name, path in self.voice_presets.items():
            if name.lower() in voice_lower or voice_lower in name.lower():
                return path

        # Default
        if self.voice_presets:
            first = list(self.voice_presets.values())[0]
            print(f"Warning: Voice '{voice_name}' not found, using default.")
            return first

        raise HTTPException(status_code=404, detail="No voices available")

    def load_voice_data(self, voice_path: str):
        if voice_path.endswith(".pt"):
            import torch  # Fallback if torch is present, though we prefer numpy

            return torch.load(voice_path, map_location="cpu", weights_only=False)

        data = np.load(voice_path)
        result = {}
        for key in data.files:
            parts = key.split("/")
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = data[key]
        return result


model_wrapper = VibeVoiceModelWrapper()

# ============================================================================
# API
# ============================================================================

app = FastAPI(title="VibeVoice HTTP Server")

APP_DIR = pathlib.Path(__file__).parent
FRONTEND_FILE = APP_DIR / "frontend" / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpeechRequest(BaseModel):
    model: str = "vibevoice"
    input: str
    voice: Optional[str] = "Wayne"
    response_format: Optional[str] = "wav"
    cfg_value: Optional[float] = 1.5
    inference_timesteps: Optional[int] = (
        5  # Used to be 10 in example, usage here maps to dpm steps?
    )
    # prompt_wav_path and prompt_text for custom creation are NOT supported fully in numpy-only yet without encoder
    # but we keep fields for compatibility or future use
    prompt_wav_path: Optional[str] = None
    prompt_text: Optional[str] = None


class PlaybackRequest(SpeechRequest):
    show_progress: Optional[bool] = True


@dataclass
class GenerationJob:
    request: SpeechRequest
    output_queue: queue.Queue
    cancel_event: threading.Event
    job_id: int


GENERATION_QUEUE = queue.Queue(maxsize=1)
CURRENT_JOB: Optional[GenerationJob] = None
JOB_COUNTER = 0


class AudioStreamerAdapter:
    def __init__(self, queue: queue.Queue):
        self.queue = queue

    def put(self, chunk, _id=None):
        self.queue.put(chunk)

    def end(self, _id=None):
        pass  # We handle end via None sentinel in worker


def normalize_text(text):
    text = ftfy.fix_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_inputs_numpy(processor, text, all_prefilled_outputs):
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors=None,
        return_attention_mask=True,
    )
    tts_text_ids = np.array(inputs["tts_text_ids"], dtype=np.int64)
    tts_lm_input_ids = np.array(inputs["tts_lm_input_ids"], dtype=np.int64)

    if tts_text_ids.ndim == 1:
        tts_text_ids = tts_text_ids[np.newaxis, :]
    if tts_lm_input_ids.ndim == 1:
        tts_lm_input_ids = tts_lm_input_ids[np.newaxis, :]
    return tts_text_ids, tts_lm_input_ids


def generation_worker():
    global CURRENT_JOB
    while True:
        try:
            job = GENERATION_QUEUE.get()
            CURRENT_JOB = job
            try:
                # 1. Load context
                voice_path = model_wrapper.get_voice_path(job.request.voice)
                all_prefilled_outputs = model_wrapper.load_voice_data(voice_path)

                # 2. Prepare Inputs
                text_input = normalize_text(job.request.input)
                tts_text_ids, tts_lm_input_ids = prepare_inputs_numpy(
                    model_wrapper.processor, text_input, all_prefilled_outputs
                )

                # 3. Generate
                streamer = AudioStreamerAdapter(job.output_queue)

                def stop_check():
                    return job.cancel_event.is_set()

                model_wrapper.generator.generate(
                    tts_text_ids=tts_text_ids,
                    tts_lm_input_ids=tts_lm_input_ids,
                    all_prefilled_outputs=all_prefilled_outputs,
                    cfg_scale=job.request.cfg_value,
                    max_length=2048,  # Could expose this
                    return_speech=True,
                    verbose=False,
                    audio_streamer=streamer,
                    stop_check_fn=stop_check,
                )

            except Exception as e:
                traceback.print_exc()
                job.output_queue.put(e)
            finally:
                job.output_queue.put(None)  # Sentinel
                CURRENT_JOB = None
                GENERATION_QUEUE.task_done()
        except Exception as e:
            time.sleep(1)


@app.on_event("startup")
async def startup_event():
    try:
        model_wrapper.load()
        worker = threading.Thread(target=generation_worker, daemon=True)
        worker.start()
    except Exception as e:
        print(f"Failed to start server: {e}")
        # raise e


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    if FRONTEND_FILE.exists():
        async with aiofiles.open(FRONTEND_FILE, mode="r") as f:
            return HTMLResponse(content=await f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.get("/health")
async def health_check():
    return {"status": "ok", "loaded": model_wrapper.generator is not None}


@app.get("/voices")
async def list_voices():
    return {"voices": model_wrapper.available_voices}


@app.post("/v1/audio/speech/cancel")
async def cancel_generation():
    global CURRENT_JOB
    if CURRENT_JOB:
        CURRENT_JOB.cancel_event.set()
        return {"status": "cancelled"}
    return {"status": "no_job_running"}


async def poll_queue(q):
    while True:
        try:
            item = q.get_nowait()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        except queue.Empty:
            await asyncio.sleep(0.01)


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    global JOB_COUNTER
    JOB_COUNTER += 1

    q = queue.Queue(maxsize=1024)
    evt = threading.Event()
    job = GenerationJob(request, q, evt, JOB_COUNTER)

    try:
        GENERATION_QUEUE.put_nowait(job)
    except queue.Full:
        raise HTTPException(429, "Busy")

    chunks = []
    try:
        async for chunk in poll_queue(q):
            chunks.append(chunk)
    except Exception as e:
        raise HTTPException(500, str(e))

    if not chunks:
        raise HTTPException(500, "No audio generated")

    audio = np.concatenate(chunks)

    # Encode
    buffer = io.BytesIO()
    fmt = request.response_format.lower()

    # Normalize to -1..1 just in case
    audio = np.clip(audio, -1.0, 1.0)

    if fmt == "wav":
        soundfile.write(buffer, audio, 24000, format="WAV")
        media_type = "audio/wav"
    elif fmt == "flac":
        soundfile.write(buffer, audio, 24000, format="FLAC")
        media_type = "audio/flac"
    elif PYDUB_AVAILABLE:
        # Pydub conversion
        pcm = (audio * 32767).astype(np.int16)
        seg = AudioSegment(pcm.tobytes(), frame_rate=24000, sample_width=2, channels=1)
        seg.export(buffer, format=fmt)
        media_type = f"audio/{fmt}"
    else:
        raise HTTPException(
            501, f"Format {fmt} requiring pydub/ffmpeg not supported in this env"
        )

    buffer.seek(0)
    return Response(content=buffer.getvalue(), media_type=media_type)


@app.post("/v1/audio/speech/stream")
async def stream_speech(request: SpeechRequest):
    global JOB_COUNTER
    JOB_COUNTER += 1

    q = queue.Queue(maxsize=1024)
    evt = threading.Event()
    job = GenerationJob(request, q, evt, JOB_COUNTER)

    try:
        GENERATION_QUEUE.put_nowait(job)
    except queue.Full:
        raise HTTPException(429, "Busy")

    async def gen():
        try:
            async for chunk in poll_queue(q):
                pcm = (chunk * 32767).astype(np.int16)
                yield pcm.tobytes()
        finally:
            evt.set()

    return StreamingResponse(
        gen(), media_type="application/octet-stream", headers={"X-Sample-Rate": "24000"}
    )


@app.post("/v1/audio/speech/playback")
async def playback(request: PlaybackRequest):
    global JOB_COUNTER, CURRENT_JOB
    JOB_COUNTER += 1
    q = queue.Queue(maxsize=1024)
    evt = threading.Event()
    job = GenerationJob(request, q, evt, JOB_COUNTER)

    try:
        GENERATION_QUEUE.put_nowait(job)
    except queue.Full:
        raise HTTPException(429, "Busy")

    if not sd.query_devices():
        raise HTTPException(500, "No audio device")

    try:
        with sd.OutputStream(samplerate=24000, channels=1, dtype="float32") as stream:
            async for chunk in poll_queue(q):
                stream.write(chunk.astype(np.float32))
    except Exception as e:
        raise HTTPException(500, str(e))

    return {"status": "done"}


def start_server():
    uvicorn.run("vibevoiceane.server:app", host="0.0.0.0", port=8000, log_level="info")
