"""Render the frozen full-video result; never retrain or select on this figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

report = json.loads(Path("docs/results/multimodal/final.json").read_text())
passed = report["success_rule_met"]
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
fig.patch.set_facecolor("#f6f7f4")
names = ["history_shrunk", "context", "full"]
labels = [
    "Histórico do criador\n(ajustado pela categoria)",
    "Histórico + categoria\n+ duração",
    "Análise completa\nimagem + fala + edição\n+ título",
]
values = [report["metrics"][name]["mae_pp"] for name in names]
bars = left.barh(
    np.arange(3),
    values,
    color=["#97a59e", "#517367", "#2e7393" if passed else "#b35840"],
    height=0.5,
)
left.set_yticks(np.arange(3), labels)
left.invert_yaxis()
left.set_xlim(0, max(values) * 1.3)
left.set_title("Erro médio de retenção\nMENOR É MELHOR", loc="left", weight="bold", pad=15)
left.set_xlabel("Pontos percentuais")
for bar, value in zip(bars, values, strict=True):
    left.text(
        value + max(values) * 0.025,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f} pp".replace(".", ","),
        va="center",
        weight="bold",
    )

comparisons = [report["full_model_comparisons"][name] for name in names[:2]]
for i, result in enumerate(comparisons):
    value, low, high = (result[key] for key in ("mean_error_reduction_pp", "ci95_low", "ci95_high"))
    color = "#2e7393" if low > 0 else "#b35840"
    right.plot([low, high], [i, i], color=color, linewidth=3)
    right.scatter([value], [i], color=color, s=70, zorder=3)
    right.annotate(
        f"{value:+.1f} pp\nIC 95%: {low:+.1f} a {high:+.1f}".replace(".", ","),
        (value, i),
        xytext=(0, 13),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )
right.set_yticks([0, 1], ["Sobre o histórico", "Sobre contexto + duração"])
right.set_ylim(-0.6, 1.7)
right.invert_yaxis()
right.axvline(0, color="#70756e", linestyle="--", linewidth=1)
right.set_title("Quanto mudou o erro\nPOSITIVO É MELHOR", loc="left", weight="bold", pad=15)
right.set_xlabel("Redução do erro (pp)\nIntervalo de 95% por criador")
for axis in (left, right):
    axis.set_facecolor("#f6f7f4")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.15)
    axis.set_axisbelow(True)
headline = (
    "A análise completa superou as duas referências neste teste"
    if passed
    else "A análise completa não confirmou vantagem neste teste"
)
fig.suptitle(
    f"{headline}\n{report['test_videos']} vídeos reservados · {report['test_creators']} criadores",
    fontsize=15,
    weight="bold",
    color="#273f36",
)
fig.supxlabel(
    "Exemplo de erro: prever 60% assistido quando o real foi 50% → 10 pontos percentuais.\nTsinghua ShortVideo: retenção no painel de observação. Não mede visualizações/curtidas após publicação.",
    fontsize=10,
)
path = Path("docs/assets/full-video-results.png")
fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
print(path)
