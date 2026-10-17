"""Reusable full-content analyzer: pinned visual/language encoders, ASR and editing.

No popularity targets enter this module. Models describe content; their predictive
value is established separately, on unseen videos with time-correct histories.
"""

import hashlib
import importlib.metadata
import io
import json
import subprocess
import sys
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from galaxies.io import sha256_file, write_json
from galaxies.media import _run

MODEL_SPECS = {
    "vision": {
        "repo": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "revision": "1a25a446712ba5ee05982a381eed697ef9b435cf",
        "files": ["open_clip_model.safetensors", "open_clip_config.json", "README.md"],
    },
    "speech": {
        "repo": "Systran/faster-whisper-small",
        "revision": "536b0662742c02347bc0e980a01041f333bce120",
        "files": ["*.json", "model.bin", "vocabulary.txt", "README.md"],
    },
    "language": {
        "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "files": ["*.json", "*.safetensors", "tokenizer*", "sentencepiece*", "*.txt", "README.md"],
    },
}


def provision_weights(root: Path):
    from huggingface_hub import snapshot_download

    for name, spec in MODEL_SPECS.items():
        snapshot_download(
            spec["repo"],
            revision=spec["revision"],
            allow_patterns=spec["files"],
            ignore_patterns=["onnx/*", "openvino/*"],
            local_dir=root / name,
            max_workers=2,
        )
    write_json(root / "model-specs.json", MODEL_SPECS)


def frame_times(duration):
    """Cover the opening and the whole span; never imply continuous semantic vision."""
    if duration <= 0:
        raise ValueError("Duração precisa ser positiva.")
    return sorted(
        set(
            [
                *np.linspace(0, max(0, duration - 0.2), 12),
                *[t for t in (0.5, 1.5, 2.5) if t < duration],
            ]
        )
    )


def audio_statistics(samples, sample_rate=16000):
    window = int(sample_rate * 0.04)
    usable = len(samples) // window * window
    if not usable:
        return {
            "audio_rms": 0.0,
            "audio_dynamic_range": 0.0,
            "silence_window_ratio": 1.0,
            "audio_zero_crossing": 0.0,
        }
    windows = samples[:usable].reshape(-1, window)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    return {
        "audio_rms": float(np.sqrt(np.mean(samples**2))),
        "audio_dynamic_range": float(np.quantile(rms, 0.9) - np.quantile(rms, 0.1)),
        "silence_window_ratio": float(np.mean(rms < 0.01)),
        "audio_zero_crossing": float(np.mean(np.signbit(samples[1:]) != np.signbit(samples[:-1]))),
    }


class DeepMediaAnalyzer:
    """Load each encoder once and preserve resumable, content-addressed features."""

    def __init__(self, weights: Path, cache: Path, device="auto"):
        import open_clip
        import torch
        from faster_whisper import WhisperModel
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        self.device = (
            ("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
        )
        self.cache = cache
        self.gpu_lock = Lock()
        self.cache_lock = Lock()
        cache.mkdir(parents=True, exist_ok=True)
        self.vision, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=str(weights / "vision/open_clip_model.safetensors"),
            device=self.device,
        )
        self.vision.eval()
        self.language = SentenceTransformer(
            str(weights / "language"), device=self.device, local_files_only=True
        )
        self.speech = WhisperModel(
            str(weights / "speech"),
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
            num_workers=3,
            local_files_only=True,
        )
        self.identity = {
            "models": MODEL_SPECS,
            "device": self.device,
            "weights": {
                str(p.relative_to(weights)): sha256_file(p)
                for p in sorted(weights.rglob("*"))
                if p.is_file()
                and p.suffix in (".safetensors", ".bin", ".json", ".model", ".txt")
                and ".cache" not in p.parts
                and p.name != "manifest.json"
            },
            "packages": {
                n: importlib.metadata.version(n)
                for n in (
                    "open-clip-torch",
                    "torch",
                    "faster-whisper",
                    "sentence-transformers",
                    "scenedetect",
                    "opencv-python",
                    "numpy",
                )
            },
            "code": {
                n: sha256_file(Path(__file__).with_name(n))
                for n in ("deep_media.py", "video_structure.py", "media.py")
            },
            "ffmpeg_version": _run(["ffmpeg", "-version"]).decode().splitlines()[0],
            "speech_options": {
                "cpu_threads": 2,
                "num_workers": 3,
                "beam_size": 3,
                "temperature": 0,
                "vad_filter": True,
                "condition_on_previous_text": False,
            },
            "frame_sampling": "12 equally spaced frames plus 0.5, 1.5 and 2.5 seconds when present",
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()
        write_json(cache / "extractor.json", {"fingerprint": self.fingerprint, **self.identity})

    def text_embedding(self, text):
        if not text.strip():
            dimension = getattr(
                self.language,
                "get_embedding_dimension",
                self.language.get_sentence_embedding_dimension,
            )
            return [0.0] * dimension()
        tokens = self.language.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        chunks = [
            self.language.tokenizer.decode(tokens[i : i + 110]) for i in range(0, len(tokens), 100)
        ]
        with self.gpu_lock:
            vectors = self.language.encode(
                chunks, batch_size=16, normalize_embeddings=True, show_progress_bar=False
            )
        vector = vectors.mean(axis=0)
        return (vector / max(np.linalg.norm(vector), 1e-12)).tolist()

    def analyze(self, path: Path, caption=""):
        import torch

        path = path.resolve()
        if not path.is_file() or path.stat().st_size > 150 * 1024**2:
            raise ValueError("Vídeo ausente ou maior que 150 MB.")
        digest = sha256_file(path)
        key = hashlib.sha256(f"{digest}:{caption}:{self.fingerprint}".encode()).hexdigest()
        cache = self.cache / f"{key}.json"
        with self.cache_lock:
            if cache.exists():
                return {**json.loads(cache.read_text()), "cache_hit": True}
        metadata = json.loads(
            _run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-show_format",
                    "-show_streams",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )
        duration = float(metadata.get("format", {}).get("duration", 0))
        streams = metadata.get("streams", [])
        if not 0 < duration <= 180 or not any(s.get("codec_type") == "video" for s in streams):
            raise ValueError("A análise requer vídeo de até 180 segundos.")
        if any(s.get("width", 0) * s.get("height", 0) > 3840 * 2160 for s in streams):
            raise ValueError("Resolução maior que 4K.")
        structure = subprocess.run(
            [sys.executable, "-m", "galaxies.video_structure", str(path)],
            check=True,
            capture_output=True,
            timeout=240,
        )
        structure = json.loads(structure.stdout)
        times, frames = [], []
        for at in frame_times(duration):
            raw = _run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-ss",
                    str(at),
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=384:384:force_original_aspect_ratio=decrease",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "pipe:1",
                ]
            )
            if raw:
                frames.append(self.preprocess(Image.open(io.BytesIO(raw)).convert("RGB")))
                times.append(float(at))
        if not frames:
            raise ValueError("Não foi possível amostrar frames.")
        with self.gpu_lock, torch.inference_mode():
            vision = self.vision.encode_image(torch.stack(frames).to(self.device))
            vision = torch.nn.functional.normalize(vision, dim=-1).cpu().numpy()
        early = vision[np.array(times) <= 3]
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        segments, speech_text, speech_features, language = [], "", {}, None
        if has_audio:
            raw = _run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-i",
                    str(path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                timeout=120,
            )
            samples = np.frombuffer(raw, dtype="<f4").copy()
            speech_features = audio_statistics(samples)
            stream, info = self.speech.transcribe(
                samples,
                beam_size=3,
                temperature=0,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            language = info.language
            for segment in stream:
                start, end = max(0.0, segment.start), min(duration, segment.end)
                if end > start:
                    segments.append(
                        {
                            "start": start,
                            "end": end,
                            "text": segment.text.strip(),
                            "avg_logprob": segment.avg_logprob,
                            "no_speech_probability": segment.no_speech_prob,
                        }
                    )
            speech_text = " ".join(s["text"] for s in segments)
        speech_features.update(
            {
                "has_audio": int(has_audio),
                "has_transcript": int(bool(speech_text)),
                "speech_characters_per_second": len(speech_text.replace(" ", "")) / duration,
                "speech_segment_coverage": min(
                    1.0, sum(s["end"] - s["start"] for s in segments) / duration
                ),
                "first_speech_s": segments[0]["start"] if segments else duration,
            }
        )
        result = {
            "content_hash": digest,
            "extractor_fingerprint": self.fingerprint,
            "duration_s": duration,
            "frame_times": times,
            "visual_mean": vision.mean(axis=0).tolist(),
            "visual_opening": (early.mean(axis=0) if len(early) else vision[0]).tolist(),
            "visual_variation": vision.std(axis=0).tolist(),
            "caption_embedding": self.text_embedding(caption),
            "speech_embedding": self.text_embedding(speech_text),
            "transcript": speech_text,
            "transcript_segments": segments,
            "language_detected": language,
            "speech_features": speech_features,
            "editing": structure,
            "warnings": [
                "ASR and cuts are automatic estimates; not manually validated.",
                "Visual semantics use sampled frames; the editing detector decodes all frames.",
                "No emotion, identity, product-quality or virality labels are inferred.",
            ],
            "cache_hit": False,
        }
        with self.cache_lock:
            write_json(cache, result)
        return result
