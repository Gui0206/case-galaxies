"""Run the saved real models. No training, collection or new experiment."""

import argparse
import json
from pathlib import Path

from galaxies.real_service import RealCreativeRequest, RealPredictionService

# First three chronologically exported test rows; no selection by prediction quality.
EXAMPLES = ["a4TPwpWGN2I", "4f16xkWakkI", "6_ZP3BPzZxI"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(".runtime"))
    parser.add_argument("--request", type=Path, help="JSON with a new RealCreativeRequest")
    parser.add_argument("--video-id", help="Recompute one existing historical example")
    parser.add_argument("--output", type=Path, default=Path(".runtime/real-demo.json"))
    args = parser.parse_args()
    service = RealPredictionService(args.workspace)
    if args.request:
        results = [
            service.predict(RealCreativeRequest.model_validate_json(args.request.read_text()))
        ]
    else:
        results = [service.replay(i) for i in ([args.video_id] if args.video_id else EXAMPLES)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("\nMODELOS REAIS · SMTPD · views em 7 dias")
    print("Experimental: ganho sobre histórico NÃO comprovado. Nenhum treino ou coleta.\n")
    print("| Vídeo | Histórico | Modelo | Intervalo de 90% | Observado | Mídia |")
    print("|---|---:|---:|---|---:|---|")
    for r in results:
        observed = r.get("demonstration", {}).get("observed_views_7d")
        print(
            f"| {r['creative_id']} | {r['historical_baseline']:,.0f} | {r['predicted_metric']:,.0f} | {r['interval_low']:,.0f}–{r['interval_high']:,.0f} | {observed if observed is not None else 'não disponível'} | {'edição' if r['media_used_for_prediction'] else 'sem mídia'} |"
        )
    print("\nImagem/fala: extração disponível; correção selecionada usa somente edição.")
    print("Os exemplos não são uma nova avaliação. O teste reservado completo continua negativo.")
    print(f"Detalhes, referências históricas e explicações: {args.output.resolve()}")


if __name__ == "__main__":
    main()
