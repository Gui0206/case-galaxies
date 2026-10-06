"""Export a scientific evaluation figure from a real generated model report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("model",type=Path)
parser.add_argument("output",type=Path)
args = parser.parse_args()
report = json.loads((args.model / "report.json").read_text())
rows = json.loads((args.model / "test_predictions.json").read_text())
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig, axes = plt.subplots(1,3,figsize=(13.5,4.5),layout="constrained")
fig.set_facecolor("#f6f7f3")
label = "DADOS SINTÉTICOS · NÃO É VALIDAÇÃO REAL" if report["evidence_level"] == "synthetic_demo" else "AVALIAÇÃO RETROSPECTIVA"
fig.suptitle(f"Galaxies · evidências do modelo\n{label}",fontsize=15,color="#234e3e",fontweight="bold")
axes[0].barh(["Histórico do canal","Modelo multimodal"],[report["baseline"]["pair_accuracy"],report["metrics"]["pair_accuracy"]],color=["#d4dccb","#719b7b"])
axes[0].set_xlim(0,1); axes[0].set_xlabel("Acerto de ordenação entre pares"); axes[0].set_title("Decisão comparativa")
x=np.log1p([r["actual"] for r in rows]); y=np.log1p([r["prediction"] for r in rows])
axes[1].scatter(x,y,s=13,alpha=.35,color="#447d62")
limits=[min(x.min(),y.min()),max(x.max(),y.max())]
axes[1].plot(limits,limits,"--",color="#a2ad97",linewidth=1)
axes[1].set_xlabel("Resultado observado · log(1 + y)"); axes[1].set_ylabel("Previsão · log(1 + y)"); axes[1].set_title("Previsão versus resultado")
names=list(report["by_sector"]); cover=[report["by_sector"][n]["interval_coverage"] for n in names]
axes[2].bar(names,cover,color="#9cb591"); axes[2].axhline(report["config"]["coverage"],linestyle="--",color="#b88a51",label="Cobertura nominal")
axes[2].set_ylim(0,1.05); axes[2].set_title("Cobertura por setor"); axes[2].legend(frameon=False)
for ax in axes: ax.set_facecolor("#f6f7f3")
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=180,facecolor=fig.get_facecolor())
print(args.output)
