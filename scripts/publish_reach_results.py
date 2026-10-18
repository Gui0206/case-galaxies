"""Publish the frozen reach study, with conclusions determined by its registered gates."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galaxies.io import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runtime/reports/reach"
DATA = ROOT / ".runtime/public/beyond-views"
DEST = ROOT / "docs/results/reach"
NAMES = {
    "history": "Histórico do canal",
    "context": "Histórico + contexto",
    "content": "Histórico + contexto + título",
}


def number(value, digits=1):
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def main():
    result = read_json(RUN / "final.json")
    if result["test_predictions_sha256"] != sha256_file(RUN / "test_predictions.csv"):
        raise ValueError("Predictions differ from the frozen report")
    if result["selection_sha256"] != sha256_file(RUN / "selection.json"):
        raise ValueError("Selection changed after the final evaluation")
    selection = result["selection"]
    for key, filename in (
        ("plan_sha256", "plan.json"),
        ("development_sha256", "development.json"),
        ("predictions_sha256", "development_predictions.csv"),
    ):
        if selection[key] != sha256_file(RUN / filename):
            raise ValueError("Frozen development artifact changed")
    plan = read_json(RUN / "plan.json")
    if plan["audit"]["cohort_sha256"] != sha256_file(DATA / "cohort.parquet"):
        raise ValueError("Source cohort changed")
    DEST.mkdir(parents=True, exist_ok=True)
    for filename in (
        "plan.json",
        "development.json",
        "development_predictions.csv",
        "selection.json",
        "final.json",
        "verification.json",
        "test_predictions.csv",
    ):
        shutil.copy2(RUN / filename, DEST / filename)
    for filename in ("audit.json", "acquisition.json", "semantic.json"):
        shutil.copy2(DATA / filename, DEST / filename)
    shutil.copy2(DATA / "source/dataverse.json", DEST / "source-record.json")
    shutil.copy2(
        RUN.parent / f"reach-final-opened-{plan['audit']['cohort_sha256']}.json",
        DEST / "evaluation-opened.json",
    )
    primary = result["sections"]["primary"]
    metrics = primary["metrics"]
    recommended = selection["recommended_forecaster"]
    chosen_name = selection.get(recommended)
    chosen_spec = plan["candidates"].get(chosen_name, {})
    chosen_description = "A previsão escolhida é uma correção aprendida sobre o histórico do canal."
    if chosen_spec.get("bundle", "").endswith("_rule"):
        similarity = (
            "similaridade semântica do título, com MiniLM multilíngue congelado"
            if chosen_spec["bundle"].startswith("semantic")
            else "semelhança lexical do título, com TF-IDF ajustado só no treino"
        )
        share = 100 * chosen_spec["shrink"]
        chosen_description = (
            f"**A solução escolhida combina {number(100 - share, 0)}% do histórico habitual com {number(share, 0)}% do resultado de vídeos parecidos.** "
            f"Ela busca até cinco comparáveis entre as últimas 50 publicações maduras do mesmo canal, por {similarity}. "
            "A combinação é feita em log e depois convertida em visualizações. A proporção foi escolhida no desenvolvimento; essa regra não ajusta uma nova regressão no treino final."
        )
    elif recommended == "history":
        chosen_description = "A solução escolhida usa somente o histórico; os candidatos de conteúdo não justificaram sua complexidade no desenvolvimento."
    gain = primary["recommended_vs_history"]
    success = result["confirmed_recommended_vs_history"]
    if success:
        conclusion = f"**O preditor escolhido no desenvolvimento reduziu o erro em {number(gain['relative_reduction_pct'])}% sobre o histórico neste teste retrospectivo, atendendo ao critério registrado.**"
    elif gain["ci95"][0] > 0:
        conclusion = f"**Houve uma redução de {number(gain['relative_reduction_pct'])}% no erro sobre o histórico, com intervalo de 95% positivo, mas abaixo do objetivo registrado de 5%.**"
    else:
        conclusion = "**O novo estudo não confirmou superioridade sobre o histórico.**"
    content_conclusion = "não confirmado pelo critério registrado"
    if result["confirmed_error_gain"]:
        content_conclusion = "confirmado pelo critério registrado"
    elif all(primary[k]["ci95"][0] > 0 for k in ("content_vs_history", "content_vs_context")):
        content_conclusion = "melhoria estatisticamente positiva, mas menor que o ganho relevante definido antes do teste"
    lines = [
        "# O novo modelo supera o histórico?",
        "",
        conclusion,
        "",
        f"Foram avaliados **{number(metrics['content']['videos'], 0)} vídeos de {number(metrics['content']['channels'], 0)} canais**, publicados de 20 a 31 de agosto de 2016 e reservados até o fim da seleção. São dados novos para esta avaliação, não publicações atuais.",
        "",
        "![Erro e ordenação no período reservado](assets/reach-results.png)",
        "",
        "## A comparação que importa",
        "",
        "| Previsor | Erro médio em log por canal ↓ | Erro médio em visualizações ↓ | Entre metade e dobro do real ↑ | Acerto ao ordenar vídeos do mesmo canal ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    ranks = primary["ranking"]["macro_accuracy"]
    for name in NAMES:
        m = metrics[name]
        label = NAMES[name] + (" **— escolhido antes do teste**" if name == recommended else "")
        lines.append(
            f"| {label} | {number(m['macro_mae_log'], 3)} | {number(m['mae_views'], 0)} | {number(m['within_factor_two'] * 100)}% | {number(ranks.get(name, 0) * 100)}% |"
        )
    lines += [
        "",
        "O erro em log permite comparar canais de tamanhos diferentes; cada canal recebe o mesmo peso. A faixa de metade a dobro é ampla: prever de aproximadamente 500 a 2.000 para um resultado de 1.000 conta como dentro da faixa. Isso não é acurácia do número exato. A ordenação também dá peso igual aos canais, não a cada par.",
        "",
        f"O histórico escolhido foi `{selection['baseline']}`; contexto: `{selection['context']}`; conteúdo: `{selection['content']}`. Essas escolhas foram feitas nas duas janelas de desenvolvimento, antes de abrir o teste.",
        "",
        "## O ganho resiste à incerteza?",
        "",
        "| Comparação | Redução relativa do erro | Intervalo de 95% da redução absoluta em log |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("recommended_vs_history", "Escolhido × histórico"),
        ("content_vs_history", "Conteúdo × histórico"),
        ("content_vs_context", "Conteúdo × contexto"),
    ):
        g = primary[key]
        lines.append(
            f"| {label} | {number(g['relative_reduction_pct'])}% | [{number(g['ci95'][0], 3)}; {number(g['ci95'][1], 3)}] |"
        )
    lines += [
        "",
        "Valores positivos significam menos erro. Um intervalo que inclui zero não estabelece superioridade. Os intervalos usam 5.000 reamostragens pareadas por canal; não são 5.000 experimentos independentes.",
        "",
        f"**Ganho adicional do conteúdo sobre histórico e contexto:** {content_conclusion}. O critério exige redução de pelo menos 5% e intervalo inteiramente positivo contra as duas referências.",
        "",
        f"**Ordenação de vídeos do mesmo canal:** {'ganho confirmado neste recorte' if result['confirmed_ranking_gain'] else 'ganho não confirmado'}. Foram {number(primary['ranking']['pairs'], 0)} pares, em {number(primary['ranking']['channels'], 0)} canais. Os pares não são independentes. A comparação exige intervalo positivo contra contexto e histórico.",
        "",
        f"Ganho de ordenação do conteúdo, em pontos percentuais: sobre histórico, **{number(100 * primary['ranking']['content_vs_history_gain'])}**, IC95% [{number(100 * primary['ranking']['content_vs_history_ci95'][0])}; {number(100 * primary['ranking']['content_vs_history_ci95'][1])}]; sobre contexto, **{number(100 * primary['ranking']['content_vs_context_gain'])}**, IC95% [{number(100 * primary['ranking']['content_vs_context_ci95'][0])}; {number(100 * primary['ranking']['content_vs_context_ci95'][1])}].",
        "",
        "Os pares são vídeos publicados pelo mesmo canal na mesma semana de calendário. Podem ter datas e condições de distribuição diferentes; não são alternativas criativas testadas simultaneamente sob o mesmo orçamento.",
        "",
        "## Cinco exemplos, sem escolher os melhores",
        "",
        "São os cinco primeiros vídeos elegíveis do teste em ordem cronológica. Os números são visualizações na data de publicação e nas seis datas seguintes.",
        "",
        "| ID do vídeo | Real | Histórico | Previsor escolhido |",
        "|---|---:|---:|---:|",
    ]
    predictions = pd.read_csv(RUN / "test_predictions.csv", dtype={"category": str})
    examples = (
        predictions[(~predictions.reserved_channel) & (predictions.history_count >= 3)]
        .sort_values(["published_at", "creative_id"])
        .head(5)
    )
    for row in examples.to_dict("records"):
        lines.append(
            f"| `{row['creative_id']}` | {number(row['target'], 0)} | {number(np.expm1(row['history']), 0)} | {number(np.expm1(row[recommended]), 0)} |"
        )
    lines += [
        "",
        "## Onde funciona e onde falha",
        "",
        "| Recorte | Vídeos | Erro do histórico ↓ | Erro do conteúdo ↓ | Ganho do conteúdo sobre histórico |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "category_2": "Automotivo",
        "category_26": "Howto & Style (categoria ampla)",
        "style_title_subset": "Títulos com termos de moda/beleza (aproximação)",
        "little_history": "Canais com menos de três resultados anteriores",
        "unseen_channels": "Canais reservados, sem histórico local",
    }
    for key, label in labels.items():
        section = result["sections"][key]
        if "metrics" not in section:
            continue
        h, c = section["metrics"]["history"], section["metrics"]["content"]
        lines.append(
            f"| {label} | {number(c['videos'], 0)} | {number(h['macro_mae_log'], 3)} | {number(c['macro_mae_log'], 3)} | {number(section['content_vs_history']['relative_reduction_pct'])}% |"
        )
    chosen = metrics[recommended]
    primary_rows = predictions[(~predictions.reserved_channel) & (predictions.history_count >= 3)]
    median_target = float(primary_rows.target.median())
    absolute_warning = ""
    if chosen["mae_views"] > metrics["history"]["mae_views"]:
        absolute_warning = f"**O erro em número de visualizações aumentou {number(100 * (chosen['mae_views'] / metrics['history']['mae_views'] - 1))}% sobre o histórico.** Uma melhora no critério principal em log não implica melhora em todas as escalas."
    control_summary = ""
    if "semantic_control" in metrics:
        control_summary = (
            f"No controle congelado, o melhor modelo lexical teve erro {number(metrics['lexical_control']['macro_mae_log'], 3)}; "
            f"o melhor semântico teve {number(metrics['semantic_control']['macro_mae_log'], 3)}. "
            "Esses controles não alteram a escolha registrada nem constituem uma nova rodada de seleção."
        )
    lines += [
        "",
        "Os recortes foram definidos antes do teste. São diagnósticos; não trocamos o resultado principal pelo recorte mais favorável.",
        "",
        absolute_warning,
        "",
        f"Os intervalos nominais de 90% do previsor escolhido cobriram **{number(chosen['coverage90'] * 100)}%** dos resultados do teste principal; a largura mediana foi **{number(chosen['median_interval_width_views'], 0)} visualizações**, para uma mediana real de **{number(median_target, 0)} visualizações**. Cobertura alta com intervalos largos ainda pode ter pouca utilidade. Coberturas por recorte estão no JSON completo.",
        "",
        "## O que foi melhorado no modelo",
        "",
        chosen_description,
        "",
        "O modelo aprende correções sobre o padrão recente do canal. Compara duração com a duração habitual, usa o texto do título e recupera até cinco publicações anteriores semelhantes, com resultados já maduros. Foram comparados 30 candidatos na rodada revisada, incluindo versões lexicais, encoder semântico multilíngue congelado, regressão regularizada, CatBoost, uma mistura fixa e regras simples de comparáveis. A primeira rodada de desenvolvimento está preservada no protocolo. Não usamos visualizações iniciais, curtidas ou retenção do próprio vídeo.",
        "",
        control_summary,
        "",
        f"Base de ajuste final dos modelos e representações que exigem treino: **{number(result['train_videos'], 0)} vídeos**. Calibração separada: **{number(result['calibration_videos'], 0)}**. A avaliação final inclui **{number(result['all_test_videos'], 0)}** vídeos ao somar histórico suficiente, pouco histórico e canais reservados. Todos os alvos do treino/calibração amadurecem antes das respectivas fronteiras. O histórico pode incorporar resultados já maduros durante o teste; os parâmetros e as regras permanecem fixos.",
        "",
        "## O que este teste ainda não prova",
        "",
        "- **Não valida Shorts atuais ou campanhas da Galaxies.** A fonte é uma amostra de 2016 de vídeos mencionados no Twitter, com estatísticas públicas e pelo menos 100 visualizações em 30 dias. Não é o catálogo completo dos canais; investimento em mídia é desconhecido.",
        "- **Não comprova ganho de imagem, fala ou edição sobre alcance.** A fonte contém metadados e séries, sem os vídeos originais. O pipeline de vídeo completo do projeto continua com o [resultado anterior de retenção](FULL_VIDEO_RESULTS.md), sem ganho confirmado.",
        "- Títulos e duração são retrospectivos. Não há prova de que sejam a versão anterior à publicação. O encoder semântico pode ter encontrado textos antigos durante seu pré-treino; a família lexical não usa esse encoder.",
        "- A semana é definida por datas, não por 168 horas. Exigimos os sete registros explícitos e não completamos ausências com zero. Consideramos o resultado disponível apenas nove dias depois da data de publicação, como convenção conservadora.",
        "- Associação preditiva não prova que editar um elemento causa mais visualizações. A aplicação continua sem promoção para uso real.",
        "",
        "## Fonte e reprodução",
        "",
        "Fonte: [Beyond Views, Wu, Rizoiu e Xie, ICWSM 2018](https://github.com/avalanchesiqi/youtube-engagement), [Harvard Dataverse, DOI 10.7910/DVN/L3UWZT](https://doi.org/10.7910/DVN/L3UWZT), arquivo 3489832, licença declarada CC0 1.0. Extraímos somente Autos & Vehicles e Howto & Style. O arquivo integral e os membros extraídos têm hashes registrados.",
        "",
        "Em um checkout novo, sem reaproveitar o teste como nova evidência:",
        "",
        "```bash",
        "uv sync --frozen --group dev --extra semantic",
        "uv run python scripts/run_reach_study.py weights",
        "uv run python scripts/download_beyond_views.py",
        "uv run python scripts/run_reach_study.py prepare",
        "uv run python scripts/run_reach_study.py encode",
        "uv run python scripts/run_reach_study.py develop",
        "uv run python scripts/run_reach_study.py final",
        "uv run python scripts/verify_reach_results.py",
        "uv run python scripts/publish_reach_results.py",
        "```",
        "",
        "O teste é de abertura única por fingerprint. Não apague o registro para retreinar e apresentar os mesmos dados como confirmação nova. Os modelos nativos e o encoder ficam em `.runtime/reports/reach/models/`; metadados brutos e pesos não foram publicados no Git.",
        "",
        "Auditoria: [protocolo e busca de bases](REVISED_STUDY.md), [fonte](results/reach/acquisition.json), [exclusões](results/reach/audit.json), [plano](results/reach/plan.json), [todos os modelos no desenvolvimento](results/reach/development.json), [previsões de desenvolvimento](results/reach/development_predictions.csv), [seleção](results/reach/selection.json), [registro da abertura final](results/reach/evaluation-opened.json), [teste completo](results/reach/final.json), [todas as previsões finais](results/reach/test_predictions.csv).",
        "",
        "A [verificação independente](results/reach/verification.json) recalcula erros e cobertura a partir de cada previsão e confere o histórico com uma segunda implementação. Ela não ajusta os modelos nem reabre a avaliação.",
    ]
    (ROOT / "docs/REACH_RESULTS.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), facecolor="#f7f8fb")
    colors = ["#9ba7b7", "#5d82b6", "#187d71"]
    labels = ["Histórico", "+ contexto", "+ título"]
    for ax in axes:
        ax.set_facecolor("#f7f8fb")
        ax.spines[["top", "right"]].set_visible(False)
    values = [metrics[n]["macro_mae_log"] for n in NAMES]
    axes[0].bar(labels, values, color=colors, width=0.6)
    axes[0].set_title("Erro de previsão · menor é melhor", loc="left", pad=18)
    axes[0].set_ylabel("Erro absoluto em log, média por canal")
    axes[0].set_ylim(0, max(values) * 1.25)
    for i, value in enumerate(values):
        axes[0].text(
            i, value + max(values) * 0.035, number(value, 3), ha="center", fontweight="bold"
        )
    accuracy = [100 * ranks[n] for n in NAMES]
    axes[1].bar(labels, accuracy, color=colors, width=0.6)
    axes[1].set_title("Ordenação de vídeos · maior é melhor", loc="left", pad=18)
    axes[1].set_ylabel("Acertos (%), média por canal")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(50, color="#8b94a2", linestyle="--", linewidth=1)
    for i, value in enumerate(accuracy):
        axes[1].text(i, value + 2.5, f"{number(value)}%", ha="center", fontweight="bold")
    title = "Novo teste: previsão de visualizações antes de publicar"
    fig.suptitle(title, fontsize=17, fontweight="bold", x=0.06, ha="left", y=0.97)
    fig.text(
        0.06,
        0.89,
        f"{number(metrics['content']['videos'], 0)} vídeos reservados · {number(metrics['content']['channels'], 0)} canais · YouTube, agosto de 2016",
        color="#4d596c",
    )
    footer = f"Redução do erro: {number(gain['relative_reduction_pct'])}% (meta de 5% {'atingida' if success else 'não atingida'}). Ordenação: ganho {'confirmado' if result['confirmed_ranking_gain'] else 'não confirmado'}."
    fig.text(0.06, 0.08, footer, fontsize=10)
    fig.text(
        0.06,
        0.035,
        "Não valida Shorts atuais nem ganho de imagem, fala e edição. Consulte os intervalos e as limitações no relatório.",
        fontsize=9,
        color="#4d596c",
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.22, wspace=0.32)
    fig.savefig(ROOT / "docs/assets/reach-results.png", dpi=170, facecolor=fig.get_facecolor())
    print(
        json.dumps(
            {
                "report": str(ROOT / "docs/REACH_RESULTS.md"),
                "recommended": recommended,
                "confirmed": success,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
