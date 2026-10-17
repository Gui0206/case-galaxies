"""Show measured timestamps of a training video without publishing source media."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("features", type=Path)
parser.add_argument("--output", type=Path, default=Path("docs/assets/video-analysis-example.png"))
args = parser.parse_args()
record = json.loads(args.features.read_text())
duration = record["duration_s"]
cuts = record["editing"]["cut_times"]
segments = record["transcript_segments"]
fig, axis = plt.subplots(figsize=(12, 5.2), layout="constrained")
fig.patch.set_facecolor("#f6f7f4")
axis.set_facecolor("#f6f7f4")
axis.axvspan(0, min(3, duration), color="#ead9aa", alpha=0.8)
axis.scatter(record["frame_times"], [2] * len(record["frame_times"]), color="#2e7393", s=60)
axis.vlines(cuts, 0.8, 1.2, color="#b35840", linewidth=1.8)
axis.broken_barh(
    [(s["start"], s["end"] - s["start"]) for s in segments],
    (-0.2, 0.4),
    facecolors="#517367",
    edgecolors="#f6f7f4",
    linewidth=0.6,
)
axis.set_yticks(
    [0, 1, 2],
    [
        "Trechos de fala\ncom início e fim",
        "Cortes detectados\nem todos os frames",
        "Momentos usados\npara entender a imagem",
    ],
)
axis.set_xlim(-duration * 0.01, duration * 1.01)
axis.set_ylim(-0.5, 2.6)
axis.set_xlabel("Posição dentro do vídeo, em segundos")
axis.spines[["top", "right", "left"]].set_visible(False)
axis.grid(axis="x", alpha=0.15)
axis.set_axisbelow(True)
axis.legend(
    handles=[Patch(facecolor="#ead9aa", label="Abertura: primeiros 3 segundos")],
    loc="upper right",
    frameon=False,
)
fig.suptitle("O que a análise mediu em um vídeo real", fontsize=19, weight="bold", color="#273f36")
axis.set_title(
    f"{duration:.1f} s de vídeo · {record['editing']['decoded_frames']:,} frames decodificados · {len(cuts)} cortes estimados · {len(segments)} trechos transcritos".replace(
        ",", "."
    ),
    fontsize=12,
    pad=17,
)
fig.supxlabel(
    "Exemplo do período de treino, ID 13024 · Tsinghua ShortVideo\nAs marcações são estimativas automáticas. Sozinhas, não demonstram maior retenção.",
    fontsize=10,
)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180, facecolor=fig.get_facecolor())
print(args.output)
