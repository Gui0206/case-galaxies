# Conseguimos superar o histórico do canal?

**A vantagem ainda não foi confirmada.** Este é o resultado de um teste reservado com **515 vídeos de 165 canais**. Os modelos foram escolhidos antes de consultar suas visualizações finais.

![Comparação em três perguntas](assets/smtpd-results.png)

## 1. Prever o número de visualizações

| Previsão | Erro em log por canal ↓ | Se o erro do histórico é 100 ↓ | Erro de até um fator 2 |
|---|---:|---:|---:|
| Histórico do canal | 0.6868 | 100,0 | 64,7% |
| Histórico + contexto | 0.6854 | 99,8 | 65,0% |
| Modelo escolhido | 0.6856 | 99,8 | 65,4% |

A redução do erro foi **0,18% contra histórico** e **-0,03% contra contexto**. Valor negativo significa piora. Isso mede erro da previsão; não representa aumento causado nas visualizações.

O critério pré-registrado de redução do erro **não foi cumprido**. Uma diferença pequena ou incerta não será apresentada como ganho comprovado.

A incerteza foi calculada reamostrando canais inteiros, 5.000 vezes. Ganho absoluto de erro contra histórico: **0.0012**, IC95% **[-0.0260; 0.0273]**. Contra contexto: **-0.0002**, IC95% **[-0.0135; 0.0126]**. Positivo significa melhora; um intervalo que cruza zero não confirma vantagem.

### Seis exemplos com números de visualizações

| Vídeo | Histórico previa | Contexto previa | Modelo previa | Real na primeira semana |
|---|---:|---:|---:|---:|
| [oEFO0m8WA2o](https://www.youtube.com/watch?v=oEFO0m8WA2o) | 9 | 16 | 15 | 2 |
| [lws0iLwpoxI](https://www.youtube.com/watch?v=lws0iLwpoxI) | 6 | 8 | 7 | 1 |
| [TotUeumIFWo](https://www.youtube.com/watch?v=TotUeumIFWo) | 5.089 | 5.555 | 5.570 | 4.162 |
| [YHBj7uWWCts](https://www.youtube.com/watch?v=YHBj7uWWCts) | 117.766 | 109.028 | 100.027 | 102.860 |
| [pnQqGEqo6jw](https://www.youtube.com/watch?v=pnQqGEqo6jw) | 56.271 | 69.579 | 73.464 | 36.366 |
| [wp4IbbXD1lc](https://www.youtube.com/watch?v=wp4IbbXD1lc) | 21.363 | 23.885 | 21.306 | 11.584 |

Os exemplos são escolhidos pela ordem do hash do ID, independentemente de o modelo acertar ou errar. Os números previstos foram arredondados; a [tabela completa](results/smtpd/final/test_predictions.csv) preserva todos os 515 casos. Links podem ficar indisponíveis; a contagem real vem da coleta histórica, não do contador atual do YouTube.

## 2. Escolher entre vídeos do mesmo canal

| Previsão | Acerto médio por canal ↑ |
|---|---:|
| Histórico do canal | 47,4% |
| Histórico + contexto | 56,8% |
| Modelo escolhido | 55,8% |

Foram **726 pares de 54 canais**, comparando publicações na mesma semana. A seleção do ordenador foi separada da previsão numérica. O critério de confirmação da ordenação **não foi cumprido**.

Esses são pares observados de vídeos publicados, não um experimento causal de trocar um roteiro por outro. Acertar qual vídeo vai melhor não significa acertar seu número de views.

## 3. O vídeo completo acrescenta valor?

Comparação nos **219 vídeos completos acessíveis de até três minutos**, de **63 canais**. Todos recebem as mesmas referências. O controle de texto usa o mesmo subconjunto de treino da análise de mídia.

| Informação acrescentada | Erro na escala em que o texto vale 100 ↓ |
|---|---:|
| Texto de referência | 100,0 |
| Texto: mesmo treino | 99,3 |
| + imagem | 100,2 |
| + fala | 100,6 |
| + edição | 99,9 |
| + todas as modalidades | 100,5 |

A diferença do modelo com todas as modalidades contra texto foi **-0,5% de redução de erro**, IC95% absoluto **[-0.0183; 0.0099]**. **Este recorte é exploratório e não alcança os 500 vídeos exigidos para uma confirmação ampla.** O melhor candidato real de cada modalidade foi congelado antes do teste, mesmo quando perdera para o fallback na validação.

O áudio foi transcrito integralmente. O detector de edição decodificou todos os frames. O encoder visual usou imagens distribuídas pelo vídeo, além dos primeiros segundos. Incluímos ritmo de cortes, fala de abertura, relação título–fala e similaridades visuais com conceitos definidos previamente; não são rótulos humanos de qualidade ou viralidade.

## Por que este teste é mais próximo do case

Usamos a [SMTPD, CVPR 2025](https://arxiv.org/abs/2503.04446), com acompanhamento diário desde a publicação. Recuperamos datas e canais estáveis pela API oficial, sem usar popularidade atual. O alvo é a contagem nominal na primeira semana. Cada resultado só entra no histórico após oito dias, incluindo margem conservadora.

O treino principal tem **796 vídeos com histórico**; a regressão de diferenças também aproveita publicações anteriores de canais repetidos. A calibração usa **129 vídeos separados**. Investigamos ainda 40.781 vídeos anteriores de outros setores, com controles de peso e ausência de transferência. Tentativas que não ajudaram permanecem registradas.

Escolha para erro: `media/editing:tree3@0.25`. Referência histórica: `base/logmean_20`. Contexto: `auxiliary/aux_context0_ridge100@0.5`. Escolha para ordenação: `base/retrieval_d3_median_20@1`. Os nomes técnicos servem para reprodução; o resultado útil está nas três comparações acima.

## O que continua sem validação

- A base é selecionada, tem histórias incompletas e exclui dias ausentes e extremos de popularidade. Vídeos removidos desde a coleta não puderam ser recuperados.
- Howto & Style inclui temas além de moda e beleza. A mídia e o título podem ter mudado desde a publicação.
- Não existe alvo de curtidas no mesmo horizonte nesta fonte. Não há confirmação prospectiva em canais da Galaxies nem autorização técnica para promover o modelo como produção validada.
- Os resultados anteriores, inclusive negativos, continuam publicados: [2016](REACH_RESULTS.md), [transferência entre épocas](TRANSFER_RESULTS.md), [adaptação recente](RECENT_ADAPTATION_RESULTS.md) e [retenção com vídeo completo](FULL_VIDEO_RESULTS.md).

## Decisão e próximo investimento que faz sentido

**Não promover este modelo como superior ao histórico.** A extração funciona, mas sua utilidade preditiva não foi demonstrada. O intervalo nominal de 90% do modelo cobriu **88,9%** dos resultados; isso não substitui o critério de ganho sobre as referências.

As tentativas de ampliar texto, transferir 40.781 vídeos, ponderar a transferência e treinar diretamente para ordenação já foram implementadas e não justificaram adoção. Alterar mais hiperparâmetros com os mesmos dados de teste criaria uma comparação contaminada. Uma nova avaliação exigirá novos dados reservados.

O próximo investimento com potencial material é obter **histórico completo de canais do domínio e resultados observados em horizonte fixo**, ligados ao vídeo original, separando mídia paga, republicações e mudanças de estratégia. O repositório já tem coleta de snapshots e registro prospectivo; esses componentes podem ser reutilizados quando houver essa fonte. A base da Academy não existe e nenhuma fonte acessível resolveu essa lacuna nesta investigação. Não há evidência para prometer que outro modelo maior corrigirá o problema.

## Revisão de encerramento: o que aprendemos

**Decisão em 18/10/2026: encerrar as tentativas de modelagem nesta entrega.** Esta revisão leu resultados, código e previsões já existentes. Não executou novos treinos, extrações, coletas ou testes preditivos. Os recortes abaixo são descritivos e posteriores à abertura do teste; não alteram a seleção congelada.

### Tema × histórico já foi explorado

A implementação compara o título com publicações anteriores do mesmo canal por palavras e embeddings semânticos. Usa até cinco comparáveis e seus resultados maduros, ponderados pela similaridade, além de medidas de similaridade máxima e peso. Também houve regressão semântica de diferenças dentro do canal e combinação com contexto. Isso testa aproximações da hipótese, não todas as possíveis representações de tema ou público.

No desenvolvimento, a melhor variante de recuperação semântica teve MAE log por canal **0,6627**, contra **0,6459** do contexto. A combinação de regressão semântica e contexto chegou a **0,6335**, mas essa promessa não se confirmou no teste final. A recuperação escolhida para ordenação era **lexical**, não a variante de embeddings semânticos. Portanto, sua seleção não prova benefício específico dos embeddings.

Fontes: [seleção base](results/smtpd/development-v2/base-selection.json), [combinação e ordenação](results/smtpd/development-v2/ensemble-selection.json), implementação em `src/galaxies/reach_study.py` (`ReachEncoder`) e `src/galaxies/smtpd_study.py`.

### Pouco histórico e ordenação não revelaram uma oportunidade clara

O teste tem mediana de **8** publicações anteriores disponíveis. Não é correto presumir que a maioria dos canais dispõe de histórico farto. Todos os vídeos avaliados têm ao menos três antecedentes maduros; **canais sem histórico não foram avaliados neste teste**.

Recortes descritivos por quantidade de antecedentes na data de cada vídeo, calculados das previsões salvas. Erro absoluto em log, primeiro médio por canal dentro do recorte e depois médio entre canais:

| Antecedentes | Vídeos | Canais presentes | Histórico | Contexto | Conteúdo |
|---|---:|---:|---:|---:|---:|
| 3–5 | 200 | 116 | 0,7164 | 0,7073 | 0,7092 |
| 6–10 | 123 | 47 | 0,5971 | 0,6433 | 0,6475 |
| 11 ou mais | 192 | 30 | 0,8020 | 0,7771 | 0,7752 |

Um canal pode aparecer em mais de um recorte à medida que publica; os canais da tabela não devem ser somados. Não calculamos novos intervalos por recorte nem usamos essas fatias para escolher modelos. O pequeno ganho pontual no último grupo não é uma confirmação. No grupo de 3–5 antecedentes, conteúdo ficou ligeiramente pior que contexto. A ordenação já registrada também ficou abaixo de contexto: **55,8% contra 56,8%**, com diferença incerta.

### Incerteza não significa ganho grande oculto

O bootstrap por canal com 5.000 reamostragens já estava registrado e verificado. Dividindo seus limites pelo erro observado da referência, o intervalo do ganho relativo do modelo principal corresponde aproximadamente a **−3,78% a +3,97% contra histórico** e **−1,97% a +1,83% contra contexto**. Positivo indica redução do erro. Esta conversão usa o denominador observado fixo, não um novo bootstrap da razão.

A meta de redução de 5% ficou fora desses intervalos para **este modelo nesta população**. Benefícios pequenos continuam compatíveis com os dados. Isso não é cálculo formal de poder, não determina o tamanho de uma futura amostra e não limita o ganho de um modelo diferente. A análise multimodal de 219 vídeos continua exploratória.

### Por que encerramos

Não encontramos evidência concreta que justifique repetir tema × histórico ou ampliar a busca de hiperparâmetros nesta entrega. A hipótese já teve implementações, a vantagem de desenvolvimento não se confirmou e o teste final já foi utilizado. Mais tentativas orientadas pelo mesmo teste contaminariam a avaliação.

Preservamos a ferramenta funcional, os resultados negativos e o histórico/contexto como referências. **Ganho não demonstrado não significa que conteúdo nunca ajuda.** Não atribuiremos a falha exclusivamente à falta de dados contínuos ou ao tamanho da amostra. Uma retomada dependeria de uma hipótese materialmente diferente, desenvolvimento separado e novos dados reservados adequados ao domínio, com autorização explícita. Nada disso está em execução.

## Conferir e reproduzir

[Previsões individuais](results/smtpd/final/test_predictions.csv) · [Resultado integral](results/smtpd/final/final.json) · [Escolha congelada](results/smtpd/final/selection.json) · [Verificação independente](results/smtpd/final/verification.json) · [Protocolo e desenvolvimento](SMTPD_PROTOCOL.md).

Os números e intervalos foram recalculados diretamente das previsões. A separação temporal e a disponibilidade do histórico foram verificadas com implementação independente.

```bash
uv run python scripts/verify_smtpd_results.py
uv run python scripts/render_smtpd_results.py
```

O comando de avaliação final recusa uma segunda abertura. Para conferir o resultado, use os arquivos publicados e a verificação; não apague o registro de abertura para tratar uma repetição como dado novo.
