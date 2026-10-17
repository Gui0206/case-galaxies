# O case está no modelo e na evidência

> **Atualização de 17/10/2026:** já há [testes em duas bases públicas reais](REAL_DATA_RESULTS.md), com resultado desfavorável à hipótese de ganho consistente sobre o histórico. Leia esse relatório primeiro. Os números de desempenho apresentados nesta página são do estudo sintético anterior e não medem eficácia real.

**A entrega é uma infraestrutura de pesquisa executável. Ainda não é um modelo com eficácia comprovada em campanhas reais.** A interface permite inspecioná-la; a qualidade depende dos dados, da representação do vídeo e do teste que tenta refutar o ganho do modelo.

A proposta técnica é combinar conteúdo multimodal com contexto disponível antes da publicação, estimar desempenho em um horizonte fixo e comparar criativos sob o mesmo contexto. A família vencedora precisa ser escolhida por avaliação temporal, não por preferência por uma arquitetura.

## Como a previsão é construída

```mermaid
flowchart LR
    V["Vídeo original<br>antes de publicar"] --> X["Texto · imagem · áudio<br>rubrica auditável"]
    H["Histórico já observado<br>canal + setor"] --> B["Referência regularizada"]
    X --> F["Atributos disponíveis<br>na decisão"]
    B --> F
    F --> C["Competição temporal<br>Ridge · CatBoost · LightGBM · blend"]
    C --> P["Contagem + ranking"]
    P --> I["Calibração separada<br>intervalos + probabilidade"]
    I --> D["Comparar ou abster-se<br>com evidências"]
```

Há duas formulações concorrentes. Na absoluta, o modelo estima `log(1 + alvo)`. Na residual, estima a diferença para `log(1 + mediana regularizada do canal)` e soma essa referência na inferência. Isso oferece ao modelo a hipótese de aprender desempenho relativo ao histórico, sem tratar diferenças de alcance como se fossem apenas qualidade criativa. Não identifica um efeito causal do conteúdo.

O contexto inclui estatísticas recentes, média móvel em log com decaimento temporal, tendência, idade do histórico, tamanho do canal, setor e regime de distribuição. Resultados que ainda não eram conhecidos são excluídos de cada consulta. Pesos de treino e métricas macro equilibram a participação dos canais.

**O nível atual dos extratores importa:** texto usa hashing + SVD; os atributos básicos de vídeo/áudio são descritivos. Whisper, OpenCLIP, OCR e VLM têm adaptadores opcionais, mas não foram avaliados em vídeos reais deste domínio. Não se apresenta o núcleo lexical como um encoder semântico de fronteira. A próxima comparação de representações depende de vídeos autorizados, transcrições, amostragem temporal e anotação humana cega. Consulte [Multimodal](MULTIMODAL.md).

## O protocolo que evita escolher pelo teste

```mermaid
flowchart TD
    D["Dataset auditado + hash"] --> R["Reservar última janela de 15%"]
    R --> W["Desenvolvimento: 3 janelas futuras<br>treino crescente + validação + calibração"]
    W --> C["12 configurações<br>mesmos IDs e cortes"]
    C --> S["Congelar candidato e referência<br>configuração + dados + código"]
    S --> T["Avaliação final única<br>candidato versus contexto forte"]
    T --> G["Critérios de pesquisa<br>elegibilidade explícita"]
```

1. **Comparação justa.** As 12 configurações usam os mesmos períodos, IDs e blocos permitidos. São três referências só de contexto, três Ridge, três CatBoost, dois LightGBM com conteúdo e um blend fixo CatBoost/Ridge. Modelos de quantis continuam usando CatBoost em todas as famílias.
2. **Seleção com dois objetivos.** Mantêm-se configurações até um erro padrão abaixo do melhor acerto macro; entre elas, excluem-se as com MAE macro em log mais de 10% acima do melhor. Depois se prefere menor complexidade, menor erro e melhor interval score. O mesmo procedimento escolhe a referência só de contexto. O erro padrão entre janelas é uma heurística; a dependência temporal impede interpretá-lo como garantia estatística.
3. **Calibração isolada.** A validação controla parada antecipada. Outra partição calibra intervalos e probabilidades de comparação; os alvos de teste não participam. Setores com pelo menos 40 exemplos de calibração recebem correção própria; os demais usam a global.
4. **Diagnósticos que podem falhar.** Ablações nos mesmos IDs, três seeds adicionais, três embaralhamentos do conteúdo de treino dentro de cada canal e exclusão de canais por setor. Os três embaralhamentos são um controle de sanidade, não um teste de permutação com p-valor confiável.
5. **Teste final registrado.** `selection.json` fixa os dois candidatos antes do teste. Hashes detectam mudanças nos dados, no código de modelagem e nas bibliotecas principais. Um registro exclusivo impede repetir a avaliação final do mesmo dataset nesse workspace. É uma proteção de auditoria, não uma defesa contra adulteração deliberada.

Um treino isolado continua útil para exploração, mas **não habilita promoção para uso real**. Essa elegibilidade exige o fluxo `research → final-evaluation`, ganho versus referências, cobertura global e por setor e amostra mínima. A confirmação final testa uma escolha congelada; não procura novamente o melhor modelo.

## Execução reproduzida: somente dados sintéticos

![Competição temporal, ablação e avaliação final em dados sintéticos](assets/research.png)

Base: **2.400 posts artificiais, 24 canais, três setores**, seed do gerador `20261006`. Treinamento com seed `42`, até 180 iterações por modelo de árvores e três janelas de desenvolvimento. O processo gerador contém relações explícitas entre atributos e alvo; recuperá-las não prova que essas relações existam em vídeos reais.

| Configuração | Acerto macro no desenvolvimento | MAE macro em log |
|---|---:|---:|
| CatBoost absoluto | 79,0% | 0,331 |
| CatBoost mais profundo | 78,8% | 0,333 |
| LightGBM absoluto | 78,7% | 0,335 |
| Blend residual | 78,5% | 0,349 |
| **CatBoost residual — selecionado** | **78,4%** | **0,329** |
| Ridge mais regularizado | 77,8% | 0,442 |
| Ridge residual | 77,8% | 0,423 |
| Ridge absoluto | 77,4% | 0,410 |
| LightGBM residual | 77,0% | 0,340 |
| **Contexto LightGBM — referência** | **52,1%** | **0,493** |
| Contexto CatBoost | 51,3% | 0,493 |
| Contexto Ridge | 50,7% | 0,563 |

O vencedor não precisa liderar uma coluna isolada: o CatBoost residual ficou dentro da margem de ranking e teve menor erro macro entre os candidatos elegíveis de mesma complexidade. A regra foi fixada antes de abrir o teste desta base.

### A janela final responde outra pergunta

Nos **360 exemplos e 622 pares** finais, o candidato congelado foi comparado à referência congelada:

| Métrica final | CatBoost residual | Contexto LightGBM |
|---|---:|---:|
| Acerto entre pares | 77,2% | 46,3% |
| MAE em log | 0,316 | 0,497 |
| MAE em curtidas | 313,9 | 454,8 |
| Cobertura do intervalo nominal de 90% | 91,4% | 90,8% |
| Largura média, em curtidas | 1.385,2 | 1.868,0 |
| Interval score em log, menor é melhor | 1,843 | 2,504 |

O ganho macro de acerto foi 30,9 pontos percentuais, com intervalo bootstrap por canal de 24,8 a 36,7 pontos. Essa incerteza descreve a amostra artificial e não a eficácia no mundo real. A cobertura por setor foi 90,8% em automotivo, 89,2% em beleza e 94,2% em moda, 120 exemplos por setor. A versão permanece `eligible_for_decision: false` por sua origem sintética.

**77,2% é acerto de ordenação entre pares, não “77,2% de precisão na quantidade de curtidas”.** Para contagens, leia os erros; para intervalos, leia cobertura, largura e penalidade por resultados fora da faixa.

### O que os diagnósticos revelaram

| Diagnóstico no desenvolvimento | Resultado | Interpretação permitida |
|---|---|---|
| Só contexto → + texto → + visual/áudio → + rubrica | 48,1% → 61,9% → 61,0% → 75,6% de acerto | Mais modalidades não garantem ganho; a rubrica domina neste gerador |
| Conteúdo de treino embaralhado dentro do canal | 45,5%, 53,0%, 47,9% | Destruir a associação conteúdo/alvo reduz o sinal artificial |
| Seeds de treino adicionais | 75,9%, 73,8%, 75,6% | Há variação; a seed mais favorável não foi escolhida |
| Canais totalmente excluídos do histórico/treino | 60 exemplos; MAE log 2,549; **0% de cobertura** | A contagem e a incerteza falham gravemente nessa coorte |

O último resultado é uma limitação importante. O acerto de ranking dessa coorte, 81,0%, não compensa contagens e intervalos inadequados. **O modelo não está validado para canais sem histórico; a regra de abstenção é necessária.** Melhorar esse cenário exige uma população real de novos canais e um modelo/calibração próprios de início de operação, avaliados separadamente. O agregado de canais vistos e novos esconderia esse problema.

## O que falta para defender excelência no modelo

| Evidência necessária | Como produzir | Situação |
|---|---|---|
| Rótulos corretos e representativos | Métrica no mesmo horizonte, origem e disponibilidade verificáveis, distribuição paga separada | Duas bases públicas auditadas; falta base representativa com versões pré-publicação verificadas |
| Representação semântica útil | Comparar texto, frames/segmentos, áudio, embeddings e rubrica nos mesmos cortes; medir concordância humana e erros de extração | Adaptadores existentes; eficácia não demonstrada |
| Valor além do histórico | Superar referências simples e contexto aprendido em erro e ordenação no futuro; reportar setores e canais | Testes públicos de metadados executados; ganho consistente não demonstrado |
| Generalização para novos clientes | Canais/clientes excluídos e calibração dedicada; manter abstenção enquanto falhar | Diagnóstico implementado e falha registrada |
| Robustez a mudança temporal | Várias janelas e acompanhamento de cobertura/erro; política de recalibração | Teste público temporal executado; seleção instável e limitações de cobertura registradas |
| Utilidade antes da publicação | Registrar previsões e versão antes de publicar; avaliar resultados maduros | Registro prospectivo implementado |
| Melhoria efetiva de campanha | Experimento controlado com condições de distribuição comparáveis | Não executado |

Sem essas evidências, aumentar complexidade, trocar de LLM ou acrescentar um ensemble não permite afirmar “melhor solução possível”. A arquitetura oferece hipóteses comparáveis e rastreáveis; a escolha de produção deve seguir o resultado real, incluindo a possibilidade de um modelo mais simples vencer.

## Reproduzir e auditar

No macOS, instale `brew install libomp ffmpeg`; no Ubuntu, `sudo apt-get install libgomp1 ffmpeg`. Em um workspace novo:

```bash
uv sync --frozen --group dev
uv run galaxies demo-data .runtime/datasets/research-validation-20261006.jsonl \
  --seed 20261006 --channels 24 --posts 100
uv run galaxies research .runtime/datasets/research-validation-20261006.jsonl \
  --iterations 180 --folds 3 --seed 42
# Use o research_id impresso acima; revise o desenvolvimento antes de abrir o teste.
uv run galaxies final-evaluation .runtime/datasets/research-validation-20261006.jsonl \
  .runtime/reports/RESEARCH_ID
```

`demo-data` recusa sobrescrever uma base existente. O teste final é de uso único por fingerprint no workspace: não apague seu registro para tentar melhorar o resultado. Uma falha após a abertura exige análise do estado e deve ser documentada; o sistema não presume que o teste continue não visto.

Artefatos desta execução: [plano](results/research/plan.json), [desenvolvimento completo](results/research/development.json), [seleção congelada](results/research/selection.json), [avaliação final](results/research/final.json) e [relatório do modelo](results/research/model-report.json). A figura é gerada por `scripts/render_research.py` a partir desses JSONs.

Durante a implementação, a base inicial de 2.160 exemplos foi usada para explorar modelos e corrigir o protocolo. Seu teste foi consultado sob uma regra anterior. A execução documentada aqui usa outra base artificial de seed fixada e a regra revisada, sem reutilizar aquele teste. Isso verifica o protocolo revisado; não cria evidência independente do processo gerador. Os resultados históricos do [demo inicial](DEMO.md) foram preservados e não devem ser comparados diretamente a estes como melhora ou regressão.

## Fundamentos

- [SMTPD, CVPR 2025](https://arxiv.org/abs/2503.04446) motiva atenção ao alinhamento temporal. Popularidade inicial do próprio vídeo pertence a outro problema; este projeto exige previsão pré-publicação.
- [Scikit-learn: validação cruzada](https://scikit-learn.org/stable/modules/cross_validation.html) documenta a necessidade de respeitar estrutura temporal e grupos. Aqui os cortes também expurgam rótulos imaturos.
- [Bracher et al.: avaliação de previsões por intervalos](https://arxiv.org/abs/2005.12881) fundamenta penalizar largura e falhas de cobertura. Implementamos interval score para um nível nominal; não uma avaliação WIS com múltiplos níveis.
- [CatBoost: SHAP](https://catboost.ai/docs/en/concepts/shap-values) descreve contribuições aditivas. Atribuições explicam o score bruto em log antes do piso de contagem, não efeitos causais de editar o vídeo.
