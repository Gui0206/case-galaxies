"""Publish a plain-language SMTPD result and a compact, reproducible scientific figure."""

import hashlib
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from galaxies.io import read_json

RUN = Path(".runtime/reports/smtpd-final")
DOCS = Path("docs")


def percent(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def main():
    report = read_json(RUN / "final.json")
    selection = read_json(RUN / "selection.json")
    verification = read_json(RUN / "verification.json")
    assert verification["prediction_sha256"] == report["predictions_sha256"]
    predictions = pd.read_csv(RUN / "test_predictions.csv")
    examples = (
        predictions.assign(
            order=predictions.creative_id.map(lambda v: hashlib.sha256(v.encode()).hexdigest())
        )
        .sort_values("order")
        .head(6)
    )
    example_rows = "\n".join(
        f"| [{row.creative_id}](https://www.youtube.com/watch?v={row.creative_id}) | {round(np.expm1(row.history)):,} | {round(np.expm1(row.context)):,} | {round(np.expm1(row.content)):,} | {int(row.target):,} |".replace(
            ",", "."
        )
        for row in examples.itertuples()
    )
    destination = DOCS / "results/smtpd/final"
    destination.mkdir(parents=True, exist_ok=True)
    for name in [
        "final.json",
        "selection.json",
        "test_predictions.csv",
        "calibration.json",
        "verification.json",
        "evaluation-opened.json",
    ]:
        shutil.copyfile(RUN / name, destination / name)
    metrics = report["primary_metrics"]
    ordering = report["ranking"]
    media = report["media_ablation"]
    labels = ["Histórico do canal", "Histórico + contexto", "Modelo escolhido"]
    roles = ["history", "context", "content"]
    error = [100 * metrics[r]["macro_mae_log"] / metrics["history"]["macro_mae_log"] for r in roles]
    rank = [100 * ordering["macro_accuracy"][r] for r in roles]
    title = (
        "Ganho confirmado no erro e na ordenação"
        if report["confirmed_error_gain"] and report["confirmed_ranking_gain"]
        else "Ganho confirmado no erro de alcance"
        if report["confirmed_error_gain"]
        else "Melhor ordenação; erro de alcance sem confirmação"
        if report["confirmed_ranking_gain"]
        else "A vantagem ainda não foi confirmada"
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    fig.patch.set_facecolor("#f5f7fb")
    fig.suptitle(title, fontsize=21, fontweight="bold", x=0.04, ha="left", color="#17263c", y=0.96)
    fig.text(
        0.04,
        0.875,
        f"SMTPD · {report['primary_videos']} vídeos novos · {report['primary_channels']} canais · publicações de janeiro–fevereiro de 2024",
        fontsize=11,
        color="#536176",
    )
    for ax in axes:
        ax.set_facecolor("#f5f7fb")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e2e7ef", zorder=0)
        ax.set_axisbelow(True)
    axes[0].barh(labels, error, color=["#a9b3c2", "#6a809e", "#1769b8"], height=0.54)
    axes[0].invert_yaxis()
    axes[0].set_title(
        "1. Prever quantas visualizações\nMenor erro é melhor",
        loc="left",
        pad=20,
        fontweight="bold",
    )
    axes[0].set_xlim(0, max(120, max(error) * 1.2))
    axes[0].set_xlabel("Erro do histórico = 100")
    for i, v in enumerate(error):
        axes[0].text(v + 2, i, percent(v), va="center", fontweight="bold")
    axes[1].barh(labels, rank, color=["#a9b3c2", "#6a809e", "#1769b8"], height=0.54)
    axes[1].invert_yaxis()
    axes[1].set_title(
        "2. Escolher qual vídeo irá melhor\nMaior acerto é melhor",
        loc="left",
        pad=20,
        fontweight="bold",
    )
    axes[1].set_xlim(0, 110)
    axes[1].set_xticks([0, 25, 50, 75, 100])
    axes[1].set_xlabel(f"Acerto médio por canal (%) · {ordering['pairs']} pares")
    for i, v in enumerate(rank):
        axes[1].text(v + 2, i, percent(v) + "%", va="center", fontweight="bold")
    names = [
        "text_reference",
        "ablation_text_control",
        "ablation_image",
        "ablation_speech",
        "ablation_editing",
        "ablation_full",
    ]
    short_labels = [
        "Texto de referência",
        "Texto: mesmo treino",
        "+ imagem",
        "+ fala",
        "+ edição",
        "+ todas as modalidades",
    ]
    short_error = [
        100
        * media["metrics"][n]["macro_mae_log"]
        / media["metrics"]["text_reference"]["macro_mae_log"]
        for n in names
    ]
    axes[2].barh(
        short_labels,
        short_error,
        color=["#a9b3c2", "#6a809e", "#58a2ba", "#58a2ba", "#58a2ba", "#1769b8"],
        height=0.6,
    )
    axes[2].invert_yaxis()
    axes[2].set_title(
        "3. Imagem, fala e edição ajudam?\nExploratório · menor erro é melhor",
        loc="left",
        pad=20,
        fontweight="bold",
    )
    axes[2].set_xlim(0, max(120, max(short_error) * 1.2))
    axes[2].set_xlabel(f"Erro do texto = 100 · {media['videos']} vídeos completos")
    for i, v in enumerate(short_error):
        axes[2].text(v + 2, i, percent(v), va="center", fontweight="bold")
    fig.text(
        0.04,
        0.055,
        "Modelos escolhidos antes do teste. Confirmação de erro exige ≥5% de ganho e IC95% positivo contra histórico e contexto.\nA análise multimodal é menor e exploratória. Dados públicos selecionados; sem validação prospectiva na Galaxies.",
        fontsize=9,
        color="#536176",
        linespacing=1.7,
    )
    fig.subplots_adjust(left=0.14, right=0.97, top=0.70, bottom=0.25, wspace=0.95)
    image = DOCS / "assets/smtpd-results.png"
    fig.savefig(image, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    gain = report["content_gains"]
    rows = "\n".join(
        f"| {label} | {metrics[role]['macro_mae_log']:.4f} | {percent(value)} | {percent(100 * metrics[role]['within_factor_two'])}% |"
        for label, role, value in zip(labels, roles, error, strict=True)
    )
    rank_rows = "\n".join(
        f"| {label} | {percent(value)}% |" for label, value in zip(labels, rank, strict=True)
    )
    media_rows = "\n".join(
        f"| {label} | {percent(value)} |"
        for label, value in zip(short_labels, short_error, strict=True)
    )
    error_conclusion = (
        "O critério pré-registrado de redução do erro foi cumprido nesta população."
        if report["confirmed_error_gain"]
        else "O critério pré-registrado de redução do erro **não foi cumprido**. Uma diferença pequena ou incerta não será apresentada como ganho comprovado."
    )
    rank_conclusion = (
        "O ganho de ordenação foi confirmado nesta população."
        if report["confirmed_ranking_gain"]
        else "O critério de confirmação da ordenação **não foi cumprido**."
    )
    text = f"""# Conseguimos superar o histórico do canal?

**{title}.** Este é o resultado de um teste reservado com **{report["primary_videos"]} vídeos de {report["primary_channels"]} canais**. Os modelos foram escolhidos antes de consultar suas visualizações finais.

![Comparação em três perguntas](assets/smtpd-results.png)

## 1. Prever o número de visualizações

| Previsão | Erro em log por canal ↓ | Se o erro do histórico é 100 ↓ | Erro de até um fator 2 |
|---|---:|---:|---:|
{rows}

A redução do erro foi **{percent(gain["history"]["relative_reduction_pct"], 2)}% contra histórico** e **{percent(gain["context"]["relative_reduction_pct"], 2)}% contra contexto**. Valor negativo significa piora. Isso mede erro da previsão; não representa aumento causado nas visualizações.

{error_conclusion}

A incerteza foi calculada reamostrando canais inteiros, 5.000 vezes. Ganho absoluto de erro contra histórico: **{gain["history"]["macro_log_error_reduction"]:.4f}**, IC95% **[{gain["history"]["ci95"][0]:.4f}; {gain["history"]["ci95"][1]:.4f}]**. Contra contexto: **{gain["context"]["macro_log_error_reduction"]:.4f}**, IC95% **[{gain["context"]["ci95"][0]:.4f}; {gain["context"]["ci95"][1]:.4f}]**. Positivo significa melhora; um intervalo que cruza zero não confirma vantagem.

### Seis exemplos com números de visualizações

| Vídeo | Histórico previa | Contexto previa | Modelo previa | Real na primeira semana |
|---|---:|---:|---:|---:|
{example_rows}

Os exemplos são escolhidos pela ordem do hash do ID, independentemente de o modelo acertar ou errar. Os números previstos foram arredondados; a [tabela completa](results/smtpd/final/test_predictions.csv) preserva todos os {report["primary_videos"]} casos. Links podem ficar indisponíveis; a contagem real vem da coleta histórica, não do contador atual do YouTube.

## 2. Escolher entre vídeos do mesmo canal

| Previsão | Acerto médio por canal ↑ |
|---|---:|
{rank_rows}

Foram **{ordering["pairs"]} pares de {ordering["channels"]} canais**, comparando publicações na mesma semana. A seleção do ordenador foi separada da previsão numérica. {rank_conclusion}

Esses são pares observados de vídeos publicados, não um experimento causal de trocar um roteiro por outro. Acertar qual vídeo vai melhor não significa acertar seu número de views.

## 3. O vídeo completo acrescenta valor?

Comparação nos **{media["videos"]} vídeos completos acessíveis de até três minutos**, de **{media["channels"]} canais**. Todos recebem as mesmas referências. O controle de texto usa o mesmo subconjunto de treino da análise de mídia.

| Informação acrescentada | Erro na escala em que o texto vale 100 ↓ |
|---|---:|
{media_rows}

A diferença do modelo com todas as modalidades contra texto foi **{percent(media["full_vs_text"]["relative_reduction_pct"])}% de redução de erro**, IC95% absoluto **[{media["full_vs_text"]["ci95"][0]:.4f}; {media["full_vs_text"]["ci95"][1]:.4f}]**. **Este recorte é exploratório e não alcança os 500 vídeos exigidos para uma confirmação ampla.** O melhor candidato real de cada modalidade foi congelado antes do teste, mesmo quando perdera para o fallback na validação.

O áudio foi transcrito integralmente. O detector de edição decodificou todos os frames. O encoder visual usou imagens distribuídas pelo vídeo, além dos primeiros segundos. Incluímos ritmo de cortes, fala de abertura, relação título–fala e similaridades visuais com conceitos definidos previamente; não são rótulos humanos de qualidade ou viralidade.

## Por que este teste é mais próximo do case

Usamos a [SMTPD, CVPR 2025](https://arxiv.org/abs/2503.04446), com acompanhamento diário desde a publicação. Recuperamos datas e canais estáveis pela API oficial, sem usar popularidade atual. O alvo é a contagem nominal na primeira semana. Cada resultado só entra no histórico após oito dias, incluindo margem conservadora.

O treino principal tem **{report["train_videos"]} vídeos com histórico**; a regressão de diferenças também aproveita publicações anteriores de canais repetidos. A calibração usa **{report["calibration_videos"]} vídeos separados**. Investigamos ainda 40.781 vídeos anteriores de outros setores, com controles de peso e ausência de transferência. Tentativas que não ajudaram permanecem registradas.

Escolha para erro: `{selection["content"]}`. Referência histórica: `{selection["history"]}`. Contexto: `{selection["context"]}`. Escolha para ordenação: `{selection["ranking_content"]}`. Os nomes técnicos servem para reprodução; o resultado útil está nas três comparações acima.

## O que continua sem validação

- A base é selecionada, tem histórias incompletas e exclui dias ausentes e extremos de popularidade. Vídeos removidos desde a coleta não puderam ser recuperados.
- Howto & Style inclui temas além de moda e beleza. A mídia e o título podem ter mudado desde a publicação.
- Não existe alvo de curtidas no mesmo horizonte nesta fonte. Não há confirmação prospectiva em canais da Galaxies nem autorização técnica para promover o modelo como produção validada.
- Os resultados anteriores, inclusive negativos, continuam publicados: [2016](REACH_RESULTS.md), [transferência entre épocas](TRANSFER_RESULTS.md), [adaptação recente](RECENT_ADAPTATION_RESULTS.md) e [retenção com vídeo completo](FULL_VIDEO_RESULTS.md).

## Decisão e próximo investimento que faz sentido

**Não promover este modelo como superior ao histórico.** A extração funciona, mas sua utilidade preditiva não foi demonstrada. O intervalo nominal de 90% do modelo cobriu **{percent(100 * metrics["content"]["coverage90"])}%** dos resultados; isso não substitui o critério de ganho sobre as referências.

As tentativas de ampliar texto, transferir 40.781 vídeos, ponderar a transferência e treinar diretamente para ordenação já foram implementadas e não justificaram adoção. Alterar mais hiperparâmetros com os mesmos dados de teste criaria uma comparação contaminada. Uma nova avaliação exigirá novos dados reservados.

O próximo investimento com potencial material é obter **histórico completo de canais do domínio e resultados observados em horizonte fixo**, ligados ao vídeo original, separando mídia paga, republicações e mudanças de estratégia. O repositório já tem coleta de snapshots e registro prospectivo; esses componentes podem ser reutilizados quando houver essa fonte. A base da Academy não existe e nenhuma fonte acessível resolveu essa lacuna nesta investigação. Não há evidência para prometer que outro modelo maior corrigirá o problema.

## Conferir e reproduzir

[Previsões individuais](results/smtpd/final/test_predictions.csv) · [Resultado integral](results/smtpd/final/final.json) · [Escolha congelada](results/smtpd/final/selection.json) · [Verificação independente](results/smtpd/final/verification.json) · [Protocolo e desenvolvimento](SMTPD_PROTOCOL.md).

Os números e intervalos foram recalculados diretamente das previsões. A separação temporal e a disponibilidade do histórico foram verificadas com implementação independente.

```bash
uv run python scripts/verify_smtpd_results.py
uv run python scripts/render_smtpd_results.py
```

O comando de avaliação final recusa uma segunda abertura. Para conferir o resultado, use os arquivos publicados e a verificação; não apague o registro de abertura para tratar uma repetição como dado novo.
"""
    (DOCS / "SMTPD_RESULTS.md").write_text(text)
    print({"report": str(DOCS / "SMTPD_RESULTS.md"), "figure": str(image)})


if __name__ == "__main__":
    main()
