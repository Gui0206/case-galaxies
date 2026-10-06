# Como saber se o modelo ajuda

**Unidade de decisão:** escolher entre criativos para o mesmo canal, setor, horizonte e contexto. Dados observacionais permitem avaliar capacidade preditiva; não provam aumento causal de resultado.

## 1. Alvo e referências

As famílias Ridge, CatBoost, LightGBM e blend disputam a predição central em `log(1 + contagem)`, diretamente ou como resíduo relativo à mediana regularizada do canal. A inversão por `expm1` produz uma estimativa central nessa escala; não se deve interpretá-la automaticamente como média aritmética da contagem.

O alvo padrão da demonstração é curtidas em sete dias. O contrato aceita outros horizontes de 1 a 90 dias, desde que cada modelo use um perfil único.

Baselines: média e mediana das últimas 20 publicações do canal com resultados já conhecidos. Sem histórico, usa-se a mediana do setor disponível naquele instante. As features de contexto também incluem uma mediana regularizada que combina canal e setor com peso `n / (n + 5)`.

## 2. Partição temporal

O esquema abaixo descreve o ajuste final. Antes dele, o comando `research` compara 12 configurações em três janelas de desenvolvimento com treino crescente, usando os mesmos IDs. Reserva-se a última janela original para uma avaliação única após congelar a seleção. A regra combina acerto macro, MAE macro em log e complexidade; o [case de modelagem](MODEL_CASE.md) detalha o protocolo e seus resultados.

```mermaid
flowchart LR
    T["Treino · 55%<br>ajustar encoder e modelo"] --> V["Validação · 15%<br>parada antecipada"]
    V --> C["Calibração · 15%<br>intervalos e pares"]
    C --> E["Teste · 15%<br>relatório final"]
```

As proporções determinam cortes por datas distintas de publicação, não sorteio de linhas. Alvos que amadurecem após o próximo corte são expurgados. Portanto, as contagens finais não precisam refletir exatamente essas porcentagens.

PCA visual e SVD de texto são ajustadas só no treino. Duplicatas exatas são removidas; grupos de campanha são mantidos apenas na primeira partição em que aparecem. A lista de IDs e os cortes ficam nos artefatos.

O histórico é consultado ponto a ponto: publicar antes do alvo não basta; o resultado também precisa ter sido observado antes da decisão. No backtest, contexto pode incorporar resultados anteriores já conhecidos durante o período de teste, sem retreinar pesos.

## 3. O que cada métrica responde

| Métrica | Pergunta |
|---|---|
| Métricas macro por canal | O resultado se mantém quando canais com muitos posts deixam de dominar a média? |
| MAE em log | Quanto erra em uma escala menos dominada por virais? |
| MAE | Qual o erro absoluto em unidades da métrica? |
| MAPE em alvos positivos | Qual o erro relativo? Quantos zeros ficaram de fora? |
| Spearman intra-canal | Ordena conteúdos dentro do mesmo canal? |
| Acerto par a par | Qual fração dos pares históricos tem a ordem correta? |
| Arrependimento de seleção | Quanto o escolhido ficou abaixo do melhor resultado observado no grupo? |
| Cobertura e largura | Quantos resultados cabem no intervalo e quão informativo ele é? |
| Interval score em log | O intervalo equilibra largura e penalidade por resultados fora dos limites? |
| Macro F1 de faixas | Distingue desempenho abaixo, típico, acima e excepcional? |
| Brier de pares | As probabilidades de comparação têm boa qualidade probabilística? |

Pares e grupos de escolha usam o mesmo canal e janelas fixas de 14 dias. Empates verdadeiros são excluídos do acerto par a par; empate previsto vale 0,5. Janelas históricas ainda diferem de criativos randomizados em uma campanha real.

As faixas relativas usam limites fixos 0,75×, 1,25× e 3× a referência do canal. “Excepcional” é relativo; não é uma definição universal de viralidade.

## 4. Intervalos e probabilidades

O CatBoost MultiQuantile estima limites inferior e superior em log. As estimativas são ordenadas e abrangem a previsão central. Na partição de calibração, calcula-se `score = max(limite_inferior - y, y - limite_superior)`. O ajuste usa a ordem `ceil((n + 1) × cobertura)`; amostras insuficientes são rejeitadas. Ajustes negativos são limitados a zero, evitando estreitamento adicional.

Isso não elimina dependência temporal ou mudança de distribuição. A cobertura reportada é empírica; uma cobertura global adequada pode esconder falhas por setor. A largura e o interval score devem sempre acompanhar a cobertura. Com ao menos 40 observações de calibração no setor, usa-se sua correção própria; caso contrário, a global. Não se reivindica garantia de cobertura condicional sob dependência temporal ou mudança de distribuição.

A probabilidade de A superar B usa calibração logística simétrica com inclinação não negativa, ajustada **na partição de calibração**, em diferenças de previsões em log. Precisa de pelo menos 30 pares e cinco canais com pares utilizáveis; canais recebem peso total igual. O relatório inclui Brier, bins de confiabilidade e acerto versus fração de comparações aceitas em vários limiares. A avaliação probabilística ocorre depois no teste. O limiar operacional de recomendação é 65%; não é uma meta de acurácia nem uma garantia universal. Ele deve ser reavaliado segundo o custo de erro na aplicação real.

## 5. Critérios de uso

Promoção real exige redução de MAE em log versus a mediana histórica, limite inferior positivo do bootstrap pareado de diferença de acerto, cobertura de pelo menos a nominal menos cinco pontos percentuais e cinco canais de teste. Além disso, exige seleção congelada verificada, ganho de acerto macro com limite bootstrap positivo e menor MAE macro versus a referência forte só de contexto, e pelo menos 30 exemplos por setor com cobertura nominal menos dez pontos percentuais. São critérios mínimos de pesquisa, não certificação de utilidade comercial. Treinos isolados permanecem inelegíveis; somente `final-evaluation` pode verificar a seleção.

O bootstrap reamostra diferenças médias por canal (1.000 amostras, seed fixa), preservando pares dependentes dentro dos canais. Ele não corrige toda dependência entre canais ou eventos comuns. A incerteza temporal precisa de avaliação adicional em produção.

Na inferência, o sistema abstém-se diante de modelo sintético, entradas simuladas, poucos resultados do canal, setor/idioma/regime não vistos, atributos fora da faixa ou maioria dos atributos ausentes. As estimativas continuam disponíveis com limitações explícitas.

## 6. Competição, diagnósticos e confirmação

```bash
uv run galaxies research minha-base.jsonl --iterations 180 --folds 3
# Revise plan.json, development.json e selection.json antes de abrir o teste.
uv run galaxies final-evaluation minha-base.jsonl .runtime/reports/RESEARCH_ID
```

O plano preserva a última janela de 15% do dataset original; só alvos já observados antes desse corte entram no desenvolvimento. Hashes fixam dados, código, bibliotecas principais e seleção. O comando final treina apenas o candidato escolhido e a referência de contexto escolhida, nos mesmos IDs finais. Um registro exclusivo por dataset impede reabrir esse teste dentro do workspace.

Ablações usam a configuração escolhida e os mesmos IDs da última janela de desenvolvimento. Controles negativos embaralham apenas o conteúdo do treino dentro de cada canal; alvos, contexto e tempos são preservados. Seeds adicionais e canais excluídos são diagnósticos, não oportunidades para selecionar a rodada mais favorável. O relatório separa canais vistos de canais novos.

`galaxies experiments DATASET` é uma suíte exploratória complementar de uma janela, mantida para especialistas por setor e CatBoostRanker. Não certifica a escolha final. Ranking produz scores de ordenação; não os interprete como contagens ou probabilidades. Compare especialistas e modelo global no mesmo setor e conjunto de IDs.

Alterar o protocolo após consultar o teste exige nova evidência futura e registro dessa mudança. Resultados sintéticos, inclusive os testes negativos, verificam a implementação e não a eficácia do conteúdo em plataformas reais.

## 7. Próxima evidência: publicação real

Receba arquivos autorizados antes de publicar; registre previsão, contexto e versão; observe o alvo no horizonte. O registro prospectivo rejeita demonstrações sintéticas e não permite sobrescrever resultados. Mudanças de horário são registradas.

Um teste prospectivo continua observacional. Demonstrar ganho de campanha requer desenho controlado, com condições comparáveis de distribuição. Não converta correlação ou ranking em promessas de ROI, CTR ou CAC.

## Referências

- [SMTPD, CVPR 2025](https://arxiv.org/abs/2503.04446): alinhamento temporal e importância da popularidade inicial. Este projeto não usa popularidade inicial do criativo como entrada.
- [Understanding Virality](https://arxiv.org/abs/2512.21402): atributos multimodais interpretáveis; seu alvo e protocolo não equivalem automaticamente aos deste projeto.
- [CatBoost SHAP](https://catboost.ai/docs/en/concepts/shap-values): contribuições aditivas do modelo, sem interpretação causal automática.
- [MAPIE](https://mapie.readthedocs.io/en/stable/): referência de incerteza conformal. A implementação aqui usa explicitamente o quantil de calibração separado, sem dependência desse pacote.

- [Bracher et al.](https://arxiv.org/abs/2005.12881): avaliação por intervalos que penaliza amplitude e falhas de cobertura. Implementamos interval score para um nível, não WIS multinível.
