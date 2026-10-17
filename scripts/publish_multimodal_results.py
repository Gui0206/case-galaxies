"""Publish frozen evidence and a plain-language report; no fitting or selection."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from galaxies.io import sha256_file, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(".runtime/public/shortvideo"))
parser.add_argument("--runs", type=Path, default=Path(".runtime/reports/multimodal"))
args = parser.parse_args()
destination = Path("docs/results/multimodal")
report = json.loads((args.runs / "final.json").read_text())
plan = json.loads((args.runs / "plan.json").read_text())
index = json.loads((args.root / "features/index.json").read_text())
if not index.get("extraction_complete"):
    raise ValueError("Extração incompleta.")
if sha256_file(args.root / "features/index.json") != plan["provenance"]["index_sha256"]:
    raise ValueError("Índice de atributos mudou depois da avaliação.")
downloads = json.loads((args.root / "download.json").read_text())
download_by_id = {x["creative_id"]: x for x in downloads["files"]}
records = []
for video, entry in index["rows"].items():
    path = args.root / "features" / entry["file"]
    if sha256_file(path) != plan["provenance"]["feature_sha256"][video]:
        raise ValueError(f"Atributos alterados: {video}")
    record = json.loads(path.read_text())
    if record["content_hash"] != download_by_id[video]["sha256"]:
        raise ValueError(f"Conteúdo diverge do arquivo baixado: {video}")
    if not all(
        0 <= s["start"] < s["end"] <= record["duration_s"] for s in record["transcript_segments"]
    ):
        raise ValueError(f"Tempos de fala inválidos: {video}")
    for key in (
        "visual_mean",
        "visual_opening",
        "visual_variation",
        "speech_embedding",
        "caption_embedding",
    ):
        if not np.isfinite(record[key]).all():
            raise ValueError(f"Atributo não finito: {video}/{key}")
    records.append(record)
audit = {
    "downloaded_videos": len(download_by_id),
    "downloaded_bytes": sum(x["bytes"] for x in download_by_id.values()),
    "download_failures": downloads["failures"],
    "extracted_videos": len(records),
    "extraction_failures": index["failures"],
    "video_hours": sum(r["duration_s"] for r in records) / 3600,
    "decoded_frames": sum(r["editing"]["decoded_frames"] for r in records),
    "semantic_image_samples": sum(len(r["frame_times"]) for r in records),
    "estimated_cuts": sum(len(r["editing"]["cut_times"]) for r in records),
    "transcript_segments": sum(len(r["transcript_segments"]) for r in records),
    "videos_with_transcript": sum(bool(r["transcript"]) for r in records),
    "automatically_detected_languages": dict(
        Counter(r["language_detected"] or "no_speech_detected" for r in records)
    ),
    "extractor_fingerprint": index["extractor_fingerprint"],
    "extraction_accuracy_against_human_labels": "not measured",
}
destination.mkdir(parents=True, exist_ok=True)
for name in (
    "plan.json",
    "development.json",
    "selection.json",
    "final.json",
    "test_predictions.csv",
):
    shutil.copyfile(args.runs / name, destination / name)
shutil.copyfile(args.root / "acquisition.json", destination / "acquisition.json")
write_json(destination / "extraction-audit.json", audit)


def number(value, decimals=1):
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


names = {
    "history_raw": "Média histórica do criador",
    "history_shrunk": "Histórico ajustado pela categoria",
    "context": "Histórico + categoria + duração",
    "caption": "Contexto + título",
    "image": "Contexto + imagem",
    "speech": "Contexto + fala/som",
    "editing": "Contexto + edição",
    "full": "Contexto + título + imagem + fala/som + edição",
}
rows = []
for name, label in names.items():
    m = report["metrics"][name]
    rows.append(
        f"| {label} | {number(m['mae_pp'], 2)} pp | {number(m['macro_creator_mae_pp'], 2)} pp | {number(m['within_10pp'] * 100)}% |"
    )
comparisons = []
for name in ("history_shrunk", "context", "caption"):
    c = report["full_model_comparisons"][name]
    comparisons.append(
        f"| {names[name]} | {number(c['mean_error_reduction_pp'], 2)} pp | {number(c['ci95_low'], 2)} a {number(c['ci95_high'], 2)} pp | {number(report['relative_mae_reduction'][name] * 100)}% |"
    )
predictions = pd.read_csv(
    args.runs / "test_predictions.csv", dtype={"creative_id": str, "channel_id": str}
)
examples = []
for r in predictions.sort_values(["first_seen", "creative_id"]).head(5).itertuples():
    examples.append(
        f"| {r.creative_id} | {number(r.target * 100)}% | {number(r.history_shrunk * 100)}% | {number(r.full * 100)}% | {number(abs(r.target - r.full) * 100)} pp |"
    )
full = report["metrics"]["full"]
history = report["metrics"]["history_shrunk"]
context = report["metrics"]["context"]
if report["success_rule_met"]:
    conclusion = "**Sim, a análise completa superou o histórico e a referência de contexto neste teste de retenção**, cumprindo o critério registrado antes de abrir o teste. Isso é evidência favorável neste painel, ainda sem demonstrar previsão de visualizações/curtidas ou eficácia em campanhas da Galaxies."
else:
    conclusion = "**Não demonstramos o ganho exigido da análise completa sobre as duas referências neste teste.** O resultado abaixo separa a comparação com o histórico simples da comparação mais difícil, que também conhece categoria e duração. A implementação funciona; sua utilidade preditiva precisa ser julgada por estes números."
training_gap = plan["provenance"]["used"] - report["train_videos"] - report["test_videos"]
pair_count = full["pair_count"]
pair_result = (
    f"{number(full['pair_accuracy'] * 100)}% em {pair_count} pares"
    if pair_count
    else "nenhum par elegível"
)
body = f"""# Imagem, fala e edição ajudam a superar o histórico?

{conclusion}

Em **{report["test_videos"]} vídeos reservados, de {report["test_creators"]} criadores**, o erro médio foi **{number(history["mae_pp"])} pontos percentuais** usando o histórico ajustado, **{number(context["mae_pp"])}** usando também categoria/duração e **{number(full["mae_pp"])}** com a análise completa.

![Comparação no teste reservado](assets/full-video-results.png)

## O que estamos prevendo

**Quanto do vídeo as pessoas assistem.** Se alguém assiste 30 segundos de um vídeo de 60 segundos, a retenção é 50%. O alvo é a média dessa fração nas observações das primeiras 24 horas desde que o vídeo aparece no painel. Repetições são limitadas a 100%.

Um erro de 10 pontos percentuais seria prever 60% assistido quando o observado foi 50%. **Menor erro é melhor.** Reduzir o erro em 10% não significa aumentar em 10% a retenção do vídeo.

A base pública oferece os arquivos completos, criadores e tempos de visualização. Ela não oferece a data de publicação necessária para validar visualizações/curtidas sete dias depois. A primeira aparição no painel não é um lançamento. Este é um estudo adicional de retenção; os [testes anteriores de metadados](REAL_DATA_RESULTS.md) mantêm seus resultados e limitações.

## O trabalho executado

| Evidência | Quantidade |
|---|---:|
| Interações auditadas na fonte | 794.053 |
| Vídeos completos baixados | {audit["downloaded_videos"]} |
| Vídeos analisados | {audit["extracted_videos"]} |
| Horas de vídeo | {number(audit["video_hours"], 2)} |
| Frames decodificados pelo detector de edição | {number(audit["decoded_frames"], 0)} |
| Imagens usadas pelo encoder visual | {number(audit["semantic_image_samples"], 0)} |
| Vídeos com fala transcrita | {audit["videos_with_transcript"]} |
| Trechos de fala com tempos | {number(audit["transcript_segments"], 0)} |
| Falhas de download / extração | {len(audit["download_failures"])} / {len(audit["extraction_failures"])} |
| Duplicatas exatas removidas | {plan["provenance"]["exact_media_duplicates_removed"]} |
| Vídeos no treino final / teste reservado | {report["train_videos"]} / {report["test_videos"]} |

Outros {training_gap} vídeos elegíveis ficaram fora do treino/teste final por atravessar o intervalo de segurança temporal. Seus resultados ainda não estariam completos no momento de treinar o modelo final. As duas janelas anteriores serviram para escolher configurações; não foi feito sorteio aleatório para misturar passado e futuro.

Usamos **OpenCLIP** para imagem, **Whisper Large v3 Turbo** para fala, um encoder semântico multilíngue para título/transcrição e **PySceneDetect com fluxo óptico** para cortes e movimento. O detector de edição decodifica todos os frames; a compreensão semântica da imagem usa amostras distribuídas, incluindo a abertura. Os modelos foram executados localmente, com versões e hashes registrados.

![O que foi extraído de um vídeo real do treino](assets/video-analysis-example.png)

## Comparação nos mesmos vídeos

| Entrada da previsão | Erro médio por vídeo ↓ | Erro equilibrado por criador ↓ | Erro até 10 pp ↑ |
|---|---:|---:|---:|
{chr(10).join(rows)}

“Contexto” significa o histórico disponível do criador e da categoria, sua variabilidade, quantidade de observações e duração. Cada combinação escolheu sua configuração usando somente o desenvolvimento. O modelo completo foi escolhido como **{report["selection"]["full"]}**; a referência de contexto, **{report["selection"]["context"]}**. Os seis modelos não usam resultados do próprio vídeo como entrada.

As linhas mostram se cada modalidade acrescenta valor à mesma referência. Não são efeitos causais: a tabela não permite prometer que inserir um corte ou trocar uma frase aumentará a retenção. Também não escolhemos outro vencedor após olhar o teste.

## A diferença é convincente?

Abaixo, um número positivo significa que a análise completa errou menos. O intervalo de 95% reamostra criadores e preserva a comparação entre previsões do mesmo vídeo.

| Referência comparada à análise completa | Redução do erro | Intervalo de 95% | Redução relativa do erro |
|---|---:|---:|---:|
{chr(10).join(comparisons)}

O critério registrado exigia reduzir o erro em pelo menos 5% contra **histórico ajustado e contexto**, com limite inferior positivo nos dois intervalos, além de pelo menos cem vídeos e cinquenta criadores. **Critério cumprido: {"sim" if report["success_rule_met"] else "não"}.** Os intervalos medem a incerteza do ganho médio, não uma garantia para cada vídeo.

Ordenação entre vídeos do mesmo criador: **{pair_result}**, considerando somente pares com pelo menos cinco pontos de diferença real. A quantidade de pares limita essa leitura; não substitui a métrica principal.

## Cinco exemplos sem escolher pelos acertos

São os primeiros cinco vídeos do teste em ordem cronológica. Os valores são retenção média, não contagem de visualizações.

| Vídeo | Observado | Histórico ajustado | Análise completa | Erro da análise |
|---|---:|---:|---:|---:|
{chr(10).join(examples)}

Todas as previsões estão no [arquivo do teste](results/multimodal/test_predictions.csv). Ele permite conferir as médias, os erros individuais e os exemplos, inclusive os desfavoráveis.

## O que esta evidência ainda não resolve

- É uma semana de recomendações observadas em um painel na China, em setembro de 2022. A população não representa campanhas brasileiras nem todos os vídeos publicados.
- A amostra exige histórico e dez observações do vídeo. Não comprova desempenho para novos criadores ou vídeos com pouca exposição.
- Os arquivos são reais, mas não há garantia de que títulos e arquivos sejam versões preservadas antes da publicação. Republicações modificadas podem escapar da deduplicação.
- A transcrição e os cortes são automáticos. Sua precisão contra anotações humanas não foi medida. Maior modelo de fala não equivale a maior precisão comprovada.
- Usuários e recomendações podem gerar dependências além do criador. O bootstrap não remove seleção de audiência nem permite inferir causalidade.
- Nenhum resultado promove automaticamente o modelo a produção. O próximo teste decisivo exige arquivos originais, contexto e previsões registrados antes de publicar, com resultados medidos no horizonte do case.

## Conferir e reproduzir

O [método e os comandos](FULL_VIDEO_METHOD.md) explicam aquisição, extratores, partições e execução. O teste final já foi aberto uma vez; o workspace impede reutilizá-lo para ajustar configurações. Reproduzir esta execução não constitui um segundo teste independente.

Arquivos auditáveis: [protocolo anterior ao teste](results/multimodal/protocol.json), [aquisição](results/multimodal/acquisition.json), [auditoria da extração](results/multimodal/extraction-audit.json), [plano com hashes e IDs](results/multimodal/plan.json), [desenvolvimento](results/multimodal/development.json), [escolha congelada](results/multimodal/selection.json) e [resultado final](results/multimodal/final.json). Os arquivos completos e os modelos treinados permanecem em `.runtime/`, fora do Git.

Fonte primária: [Tsinghua ShortVideo Dataset](https://github.com/tsinghua-fib-lab/ShortVideo_dataset), descrito na [publicação WWW 2025](https://arxiv.org/html/2502.05922v1). A extração de atributos foi implementada no repositório; nenhum resultado sintético foi apresentado como evidência deste teste.
"""
Path("docs/FULL_VIDEO_RESULTS.md").write_text(body)
print(
    json.dumps(
        {"report": "docs/FULL_VIDEO_RESULTS.md", "audit": audit}, ensure_ascii=False, indent=2
    )
)
