"""Visual report of the frozen model competition. Error bars are fold SD, not confidence intervals."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("report_folder", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
development = json.loads((args.report_folder / "development.json").read_text())
final = json.loads((args.report_folder / "final.json").read_text())
rows = list(reversed(development["selection"]["leaderboard"]))
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(
    1, 3, figsize=(15, 6), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1, 1]}
)
fig.set_facecolor("#f7f8f4")
synthetic = development["evidence_level"] == "synthetic_demo"
fig.suptitle(
    "Galaxies · seleção e verificação do modelo\n"
    + (
        "DADOS SINTÉTICOS — NÃO DEMONSTRA EFICÁCIA REAL"
        if synthetic
        else "RESULTADOS RETROSPECTIVOS"
    ),
    color="#274f40",
    fontsize=16,
    fontweight="bold",
)
axes[0].barh(
    [r["candidate"] for r in rows],
    [r["pair_accuracy_mean"] for r in rows],
    xerr=[r["pair_accuracy_sd"] for r in rows],
    color=[
        "#345f4a" if r["candidate"] == development["selection"]["selected"] else "#b0c1a8"
        for r in rows
    ],
    capsize=3,
)
axes[0].set_xlim(0.3, 1)
axes[0].set_title("Desenvolvimento · 3 janelas")
axes[0].set_xlabel("Acerto macro por canal\nBarras: desvio entre janelas; não IC")
ablations = development["diagnostics"]["ablations"]
axes[1].barh(
    ["Contexto", "+ Texto", "+ Visual/áudio", "+ Rubrica"],
    [a["metrics"]["macro_channel_pair_accuracy"] for a in ablations],
    color="#96b09a",
)
axes[1].set_xlim(0, 1)
axes[1].set_title("Ablação · última janela")
axes[1].set_xlabel("Acerto macro por canal\nMesmos IDs; não é efeito causal")
values = [final["context_metrics"]["mae_log"], final["metrics"]["mae_log"]]
axes[2].bar(["Contexto forte", "Selecionado"], values, color=["#c3ccba", "#345f4a"])
for i, value in enumerate(values):
    axes[2].text(i, value + 0.008, f"{value:.3f}", ha="center")
axes[2].set_ylim(0, max(values) * 1.25)
axes[2].set_ylabel("MAE em log · menor é melhor")
axes[2].set_title("Avaliação final")
axes[2].set_xlabel(
    f"Cobertura do selecionado: {final['metrics']['interval_coverage']:.1%}\nNominal: 90% nesta execução"
)
for ax in axes:
    ax.set_facecolor("#f7f8f4")
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
print(args.output)
