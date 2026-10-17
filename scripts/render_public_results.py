"""Render a simple, explicitly tolerant count-error comparison from real held-out data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("docs/results/public")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
fig.patch.set_facecolor("#f6f7f4")
colors = ["#517367", "#b35840", "#8d8280"]
for ax, name, title in zip(
    axes,
    ["kenjee", "trending"],
    ["Ken Jee · 30 vídeos de teste", "YouTube Trending · 377 vídeos de teste"],
    strict=True,
):
    result = json.loads((root / name / "final.json").read_text())
    candidates = ["pooled_median", "catboost_residual", result["selected"]]
    values = [result["metrics"][k]["fraction_within_factor_2_plus_one"] * 100 for k in candidates]
    bars = ax.barh(np.arange(3), values, color=colors, height=0.52)
    ax.set_yticks(
        np.arange(3),
        [
            "Histórico ajustado",
            "CatBoost residual\n(proposta anterior)",
            "Modelo escolhido\nno desenvolvimento",
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}%",
            va="center",
            weight="bold",
            fontsize=12,
        )
    ax.set_title(title, loc="left", weight="bold", fontsize=13, pad=18)
    ax.set_facecolor("#f6f7f4")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
fig.suptitle(
    "Dados reais: o modelo ainda não demonstra ganho consistente\n"
    "Previsões entre metade e dobro do resultado observado",
    fontsize=16,
    weight="bold",
    color="#273f36",
)
fig.supxlabel(
    "Tolerância ampla: resultado real de 1.000 → previsão entre aproximadamente 500 e 2.000.\n"
    "Cálculo usa contagem + 1 para tratar zeros. Teste de título/contexto; não inclui vídeo e áudio completos.",
    fontsize=10,
)
path = Path("docs/assets/real-data-results.png")
path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
print(path)
