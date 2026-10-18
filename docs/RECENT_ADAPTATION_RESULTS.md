# Aprender em dados recentes ainda não confirmou o ganho

**Teste reservado: 665 vídeos de 110 canais.** O modelo com conteúdo reduziu o erro em **2,1% contra o histórico**, mas ficou **1,3% pior que contexto + histórico**. Os intervalos de confiança incluem ausência de ganho. Não há confirmação de superioridade.

| Informação disponível antes de publicar | Erro médio por canal em log ↓ | Contra o histórico |
|---|---:|---:|
| Histórico: mediana das últimas cinco publicações maduras | 0,4657 | Referência |
| Histórico + contexto | **0,4497** | 3,4% menor |
| Histórico + contexto + significado do título | 0,4557 | 2,1% menor |

Na prática: o conteúdo acertou a faixa entre metade e o dobro das visualizações reais em **82,7%** dos vídeos; o histórico, em **81,4%**. Essa diferença pequena não basta para afirmar que o modelo oferece uma vantagem reproduzível. Os intervalos de previsão também continuam largos.

## O que mudou nesta tentativa

O teste anterior treinou em 2016 e não se transferiu para 2020–2024. Aqui treinamos diretamente na época recente, utilizando os sete países já avaliados **como desenvolvimento**. A confirmação usou quatro países novos — Índia, Japão, Coreia do Sul e Rússia — excluindo todos os 229.115 IDs observados anteriormente, inclusive os que não entraram no teste anterior.

As validações terminaram em abril de 2022. O ajuste final usou 435 vídeos com resultados maduros antes de maio; a calibração usou 39 vídeos de maio–junho. O teste começou em julho de 2022. O commit `f3cecf3` publicou o protocolo e a escolha antes da abertura dos resultados finais.

Foram comparados oito regressores, cinco intensidades de correção e oito históricos. O desenvolvimento escolheu o modelo semântico, com redução de erro de 4,2%; o desempenho reservado foi menor. Não trocamos o vencedor após observar o teste.

## Incerteza e limites

- Ganho absoluto de erro contra histórico: **0,0100**, IC95% **−0,0171 a +0,0374**. Contra contexto: **−0,0060**, IC95% **−0,0257 a +0,0134**. Positivo significa melhora. Reamostragem pareada por canal, 5.000 repetições.
- Ordenação: 39 pares de apenas 24 canais; insuficiente para confirmar uma vantagem.
- Fonte: [YouTube Trending, versão 1346](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset). Vídeos selecionados por tendências, histórias incompletas e títulos registrados depois da publicação. Não representa todos os uploads.
- Este teste não inclui imagem, fala ou edição. Não confirma nem refuta a eficácia dessas modalidades.

O teste está encerrado e o modelo não foi promovido para produção. A próxima investigação usa a SMTPD, com acompanhamento diário desde a publicação e possibilidade de recuperar os vídeos completos.

## Conferir os números

[Resultado integral](results/modern-adaptation/final.json) · [Previsões individuais](results/modern-adaptation/test_predictions.csv) · [Verificação independente](results/modern-adaptation/verification.json) · [Plano congelado](results/modern-adaptation/plan.json) · [Escolha anterior ao teste](results/modern-adaptation/selection.json).
