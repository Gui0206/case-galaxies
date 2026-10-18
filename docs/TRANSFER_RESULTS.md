# O ganho resiste a outra época?

**Não neste teste.** Em dados novos de 2020–2024, o modelo com conteúdo teve erro **3,3% maior que o histórico do canal**. A diferença não é estatisticamente conclusiva, mas certamente não comprova melhoria. A solução continua sem autorização técnica para produção.

![Resultado independente](assets/transfer-results.png)

## A comparação em linguagem simples

Pedimos ao modelo treinado em 2016 que previsse visualizações na primeira semana de vídeos de 2020–2024. Ele podia consultar resultados anteriores do canal que já estivessem completos antes da publicação. Nenhum resultado do próprio vídeo entrou na previsão. Nenhuma configuração foi escolhida usando o teste novo.

| Previsão | Erro em log por canal ↓ | Se o histórico tem erro 100 ↓ |
|---|---:|---:|
| Histórico do canal | 0,4245 | 100,0 |
| Histórico + contexto | 0,4194 | 98,8 |
| Histórico + contexto + conteúdo | 0,4386 | 103,3 |

O recorte principal tem **922 vídeos de 119 canais**, cada vídeo com pelo menos três resultados anteriores maduros. A população inteira tem 1.998 vídeos. Usamos peso igual por canal para um canal com muitos uploads não dominar a conclusão. O erro em quantidade bruta de visualizações caiu, mas essa métrica dá mais influência aos vídeos grandes e **não era o critério principal registrado**. Não a escolhemos depois para anunciar vitória.

A redução absoluta de erro sobre histórico foi **−0,0142**, com IC95% **[−0,0468; +0,0190]**, por reamostragem de canais. Contra contexto, **−0,0192**, IC95% **[−0,0404; +0,0006]**. O critério exigia redução de pelo menos 5% e intervalo positivo contra ambos: **não cumprido**.

## A ordenação também falhou

O ordenador foi escolhido separadamente, pois escolher qual vídeo irá melhor é diferente de acertar seu número de views. O treino direto de pares foi melhor em desenvolvimento e foi mantido, apesar do resultado externo ruim.

No teste havia apenas **40 pares em 18 canais** na mesma semana: histórico **49,7%**, contexto **57,2%**, conteúdo **28,8%** de acerto médio por canal. A amostra não atinge os 200 pares/30 canais exigidos. Mesmo descritivamente, o resultado é desfavorável. Não aumentamos a janela de comparação após ver o resultado.

## O que foi efetivamente melhorado

- Aprendizagem direta de ordenação dentro do canal, em vez de depender apenas de regressão de contagens.
- Comparáveis semânticos que transferem desempenho **relativo ao histórico próprio**, evitando confundir tema com tamanho de canal.
- Remoção de duração/idioma/definição ausentes na fonte externa, tanto no treino quanto no teste.
- Seleções independentes de histórico, contexto e conteúdo para previsão e ordenação.
- Teste de mudança de época, seleção, horizonte e população. A melhora de desenvolvimento não garantiu transferência.

Foram desenvolvidos 14 regressores e 10 ordenadores, além de oito históricos. A segunda rodada aumentou o treino com resultados maduros de 2016, **sem usar as publicações do teste anterior fechado de 20–31/08**. Plano e seleção foram [publicados no commit 0138217](https://github.com/Gui0206/case-galaxies/commit/0138217) antes de abrir o teste externo. As [revisões de desenvolvimento](NEXT_EVIDENCE.md) estão preservadas.

## Limites que continuam relevantes

A fonte contém vídeos que entraram em tendências, não todos os vídeos do canal. O título é o primeiro observado depois da publicação; pode ter sido alterado. O alvo externo é o primeiro snapshot nominal entre 168 e 192 horas; o treino usa sete datas de calendário. Essa diferença e a mudança de população fazem parte da dificuldade de transferência. Não conseguimos atribuir a falha a uma causa única.

Os intervalos calibrados nos dados antigos cobriram **97,0%** dos valores no teste, acima dos 90% nominais, mas a largura mediana foi **1,82 milhão de visualizações**. Cobertura alta com intervalos muito largos não torna a previsão útil.

Este teste não possui vídeo completo, fala, imagem ou curtidas. Não comprova ganho dessas modalidades nem causalidade de mudanças na edição. A [análise anterior de vídeos completos](FULL_VIDEO_RESULTS.md) permanece separada e sem ganho confirmado.

## Reproduzir e auditar

Os [resultados completos](results/transfer/final.json), [previsões individuais](results/transfer/test_predictions.csv), [seleção](results/transfer/selection.json), [plano](results/transfer/plan.json), [auditoria da base](results/transfer/audit.json) e [verificação independente](results/transfer/verification.json) são públicos. As métricas principais foram recalculadas diretamente das previsões e coincidiram com o relatório.

Em um ambiente novo, com as dependências instaladas e a fonte de 2016 preparada conforme o [protocolo anterior](REVISED_STUDY.md):

```bash
uv run python scripts/acquire_modern_sources.py
uv run python scripts/run_transfer_study.py prepare
uv run python scripts/run_transfer_study.py encode
uv run python scripts/run_transfer_study.py develop
# Após conferir e preservar plano + seleção:
uv run python scripts/run_transfer_study.py final
```

O comando final usa um registro exclusivo de abertura: no ambiente desta avaliação ele recusará uma segunda execução. Para inspecionar resultados, leia os arquivos; não apague o registro para tratar uma repetição como teste novo.
