"""Small, content-only measurements to complement unsupervised video projections.

CLIP similarities are continuous descriptors, not verified object labels or
quality/emotion judgments. Temporal speech rates interpolate ASR segments and are
explicitly estimates; no outcome value enters the descriptors.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from galaxies.io import read_json, sha256_file, write_json

PROTOTYPES = Path(".runtime/public/smtpd/visual-prototypes.json")
CONCEPTS = {
    "speaking_person": "a person speaking to the camera",
    "face_closeup": "a close-up of a human face",
    "makeup_demo": "a person applying makeup",
    "skincare_demo": "a skincare product demonstration",
    "hair_styling": "a hair styling demonstration",
    "nail_art": "a nail art demonstration",
    "fashion_outfit": "a person showing a fashion outfit",
    "clothing_closeup": "a close-up of clothing and fabric",
    "product_closeup": "a close-up of a product being displayed",
    "hands_demonstration": "hands demonstrating how to do something",
    "car_exterior": "the exterior of a car",
    "car_interior": "the interior of a car",
    "car_driving": "a car driving on a road",
    "car_repair": "a person repairing a car",
    "onscreen_text": "large written text on a video frame",
    "split_screen": "a split-screen comparison of two images",
    "food_demo": "a cooking demonstration with food",
    "animation": "an animated illustration",
    "screen_recording": "a recording of a computer screen",
    "outdoor_scene": "an outdoor landscape",
}


def prepare_prototypes():
    import open_clip
    import torch

    checkpoint = Path(".runtime/weights/vision/open_clip_model.safetensors")
    identity = {
        "model": "ViT-B-32",
        "checkpoint_sha256": sha256_file(checkpoint),
        "concepts": CONCEPTS,
        "templates": ["A photo of {}.", "A video frame of {}."],
    }
    if PROTOTYPES.exists():
        result = read_json(PROTOTYPES)
        if result["identity"] != identity:
            raise ValueError("Visual prototypes changed")
        return result
    torch.set_num_threads(4)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(checkpoint), device="cpu"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompts = [t.format(c) for c in CONCEPTS.values() for t in identity["templates"]]
    with torch.inference_mode():
        vectors = model.encode_text(tokenizer(prompts)).numpy().reshape(len(CONCEPTS), 2, -1)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=2, keepdims=True), 1e-12)
    vectors = vectors.mean(axis=1)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    result = {
        "identity": identity,
        "vectors": vectors.tolist(),
        "labels_are_not_probabilities": True,
    }
    write_json(PROTOTYPES, result)
    return result


@lru_cache(maxsize=1)
def prototypes():
    result = read_json(PROTOTYPES)
    if result["identity"]["concepts"] != CONCEPTS:
        raise ValueError("Unexpected concept definitions")
    return np.asarray(result["vectors"])


def cosine(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def derived(record, block):
    duration = record["duration_s"]
    if block == "image":
        result = {
            "opening_whole_cosine": cosine(record["visual_opening"], record["visual_mean"]),
            "visual_variation_norm": float(np.linalg.norm(record["visual_variation"])),
        }
        for field in ["visual_mean", "visual_opening"]:
            value = np.asarray(record[field])
            value = value / max(np.linalg.norm(value), 1e-12)
            score = prototypes() @ value
            result.update(
                {f"{field}:{name}": float(s) for name, s in zip(CONCEPTS, score, strict=True)}
            )
        return result
    if block == "speech":
        result = dict(record["speech_features"])
        result["caption_speech_cosine"] = cosine(
            record["caption_embedding"], record["speech_embedding"]
        )
        segments = record["transcript_segments"]
        for seconds in [3, 10]:
            end = min(seconds, duration)
            characters = sum(
                len(s["text"].replace(" ", ""))
                * max(0, min(s["end"], end) - s["start"])
                / max(s["end"] - s["start"], 1e-9)
                for s in segments
            )
            result[f"estimated_opening_{seconds}s_characters_per_second"] = characters / end
        result["mean_asr_log_probability"] = (
            float(np.mean([s["avg_logprob"] for s in segments])) if segments else 0.0
        )
        return result
    if block == "editing":
        result = dict(record["editing"]["features"])
        bins = np.histogram(record["editing"]["cut_times"], bins=np.linspace(0, duration, 6))[0]
        result.update(
            {f"cuts_in_fifth_{i}_per_second": float(v / (duration / 5)) for i, v in enumerate(bins)}
        )
        return result
    raise ValueError("Unknown media block")


if __name__ == "__main__":
    print({"prepared_concepts": len(prepare_prototypes()["vectors"])})
