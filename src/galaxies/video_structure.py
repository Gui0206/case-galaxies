"""Decode the entire video for shot boundaries and motion, in an isolated worker.

Isolation avoids loading OpenCV's and PyAV's different native FFmpeg runtimes in
the same process on macOS. Detected cuts are estimates, not human annotations.
"""

import json
import sys
from pathlib import Path

import numpy as np


def shot_statistics(cuts, duration):
    cuts = sorted({float(t) for t in cuts if 0 < t < duration})
    lengths = np.diff([0, *cuts, duration])
    return {
        "shot_count": len(lengths),
        "cuts_per_second": len(cuts) / duration,
        "early_cuts_per_second": sum(t <= 3 for t in cuts) / min(3, duration),
        "shot_mean_s": float(lengths.mean()),
        "shot_median_s": float(np.median(lengths)),
        "shot_std_s": float(lengths.std()),
        "short_shot_fraction": float(np.mean(lengths < 2)),
        "first_shot_s": float(lengths[0]),
    }


def analyze_structure(path: Path):
    import cv2
    from scenedetect import AdaptiveDetector

    cv2.setNumThreads(1)
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise ValueError("Vídeo não pôde ser aberto.")
    fps = float(video.get(cv2.CAP_PROP_FPS))
    if not 0 < fps <= 240:
        raise ValueError("Taxa de frames inválida.")
    detector = AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=max(1, int(fps * 0.3)))
    cuts, times, motion, brightness, saturation = [], [], [], [], []
    previous = None
    frame_no = 0
    stride = max(1, round(fps / 8))
    try:
        while True:
            success, frame = video.read()
            if not success:
                break
            if frame_no > 144000:
                raise ValueError("Limite de frames excedido.")
            timestamp = float(video.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if frame_no and timestamp <= times[-1]:
                timestamp = times[-1] + 1 / fps
            times.append(timestamp)
            small = cv2.resize(frame, (256, max(2, round(frame.shape[0] * 256 / frame.shape[1]))))
            cuts.extend(detector.process_frame(frame_no, small))
            if frame_no % stride == 0:
                gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
                brightness.append(float(gray.mean() / 255))
                saturation.append(
                    float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean() / 255)
                )
                if previous is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        previous, gray, None, 0.5, 2, 9, 2, 5, 1.1, 0
                    )
                    # Displacement per second in the normalized 160x90 representation.
                    magnitude = np.linalg.norm(flow, axis=2).mean() * fps / stride / 160
                    motion.append((timestamp, float(magnitude)))
                previous = gray
            frame_no += 1
    finally:
        video.release()
    if not times:
        raise ValueError("Nenhum frame decodificado.")
    duration = times[-1] + 1 / fps
    cuts = [times[index] for index in cuts if 0 <= index < len(times)]
    values = np.array([v for _, v in motion])
    early = [v for t, v in motion if t <= 3]
    return {
        "detector": "PySceneDetect AdaptiveDetector, threshold=3, min_content=15, min_scene=0.3s",
        "decoded_frames": frame_no,
        "decoded_duration_s": duration,
        "cut_times": cuts,
        "features": {
            **shot_statistics(cuts, duration),
            "motion_mean": float(values.mean()) if len(values) else 0.0,
            "motion_p90": float(np.quantile(values, 0.9)) if len(values) else 0.0,
            "early_motion_mean": float(np.mean(early)) if early else 0.0,
            "brightness_mean": float(np.mean(brightness)),
            "brightness_std": float(np.std(brightness)),
            "saturation_mean": float(np.mean(saturation)),
        },
    }


if __name__ == "__main__":
    print(json.dumps(analyze_structure(Path(sys.argv[1]).resolve())))
