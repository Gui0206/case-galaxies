<p align="center"><img src="src/galaxies/web/mark.svg" width="64" alt="Galaxies Creative Intelligence"></p>

<h1 align="center">Galaxies · Creative Intelligence</h1>
<p align="center">Prever visualizações antes de publicar.<br>Medir se o modelo supera o histórico do canal em dados reservados.</p>

<p align="center"><a href="docs/RECENT_ADAPTATION_RESULTS.md">Resultado mais recente</a> · <a href="docs/REVISED_STUDY.md">Método e busca de dados</a> · <a href="docs/FULL_VIDEO_RESULTS.md">Vídeos completos</a> · <a href="docs/MODEL_CASE.md">O case de modelagem</a> · <a href="docs/QUICKSTART.md">Começar</a> · <a href="docs/DATA.md">Preparar dados</a></p>

> **Resultado mais recente — 18/10/2026 UTC:** treinar em dados recentes ainda não confirmou a vantagem. Em **665 vídeos novos de 110 canais**, o conteúdo teve erro **2,1% menor que o histórico**, mas **1,3% maior que contexto + histórico**, com incerteza que inclui ausência de ganho. O modelo não foi promovido para produção. A investigação atual usa a [SMTPD com alvo de sete dias e vídeos completos](docs/SMTPD_PROTOCOL.md).

## Entenda o resultado em um minuto

A pergunta é: **antes de publicar, analisar o conteúdo ajuda mais que olhar o desempenho habitual do canal?** O modelo parte do histórico e ajusta a previsão com contexto, título e vídeos comparáveis. A avaliação usa resultados futuros reservados e histórias disponíveis antes de cada publicação.

| Estudo real | Resultado | O que permite concluir |
|---|---|---|
| Alcance de 2016: 27.505 vídeos de teste | Erro 3,7% menor; ordenação 60,6% vs. 49,3% | Ganho modesto nessa população; abaixo da meta de 5% de redução do erro |
| **Transferência para 2020–2024: 922 vídeos de teste** | **Erro 3,3% maior** | **O ganho não se generalizou; sem confirmação** |
| **Adaptação recente: 665 vídeos novos de teste** | **Erro 2,1% menor que histórico; 1,3% maior que contexto** | **Sem ganho confirmado** |
| Vídeos completos: retenção, 177 vídeos de teste | Sem ganho estatisticamente confirmado | Imagem, fala e edição funcionam como extração; eficácia preditiva não comprovada |

![Teste externo em outra época](docs/assets/transfer-results.png)

Leia o [teste mais recente explicado](docs/RECENT_ADAPTATION_RESULTS.md), a [transferência entre épocas](docs/TRANSFER_RESULTS.md), o [resultado de alcance anterior](docs/REACH_RESULTS.md) e o [teste de vídeos completos](docs/FULL_VIDEO_RESULTS.md). Os resultados negativos permanecem visíveis. Os dados de 2016 e 2020–2024 são públicos e selecionados; nenhum deles representa diretamente campanhas atuais da Galaxies.

As novas tentativas e a auditoria de fontes estão no [registro de pesquisa](docs/NEXT_EVIDENCE.md). A base mencionada no briefing não existe, conforme informado pelo responsável pelo projeto. O aplicativo abaixo continua sendo uma demonstração com dados sintéticos; seu funcionamento não é prova de precisão no mundo real.

## A qualidade está na avaliação do modelo

O estudo de alcance compara **30 candidatos e oito referências históricas** em duas janelas de desenvolvimento. Congela modelos, dados e código antes da avaliação final. Ele usa referências com até 50 publicações maduras, modelos residuais, texto lexical/semântico e recuperação de comparáveis. A escolha final minimiza o erro por canal, sem consultar o teste.

O [case de modelagem](docs/MODEL_CASE.md) também documenta o protocolo geral da aplicação e os estudos anteriores. A evidência sintética verifica a implementação; a eficácia real é medida nos estudos públicos, com suas limitações explícitas.

## Execute em três comandos

Requisitos: [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.12 (gerenciado pelo uv) e OpenMP para LightGBM (`brew install libomp` no macOS; `libgomp1` no Ubuntu). FFmpeg é necessário para vídeos locais.

```bash
uv sync --frozen --group dev
uv run galaxies demo
uv run galaxies serve
```

Abra **[localhost:8000](http://127.0.0.1:8000)**. A demonstração gera 2.160 posts fictícios de 24 canais, treina um modelo e ativa uma versão explicitamente sintética. Nenhuma chave de API é necessária. Os cartões ilustrados da interface representam cenários simulados, não vídeos analisados.

## Da entrada à decisão

```mermaid
flowchart LR
    V["Vídeo autorizado<br>+ contexto"] --> E["Extração<br>multimodal"]
    E --> F["Atributos + histórico<br>disponível na decisão"]
    F --> M["Modelo selecionado<br>+ quantis calibrados"]
    M --> P["Previsão + intervalo<br>+ ranking"]
    P --> X["SHAP + comparáveis<br>+ limitações"]
    X --> T["Python · HTTP · MCP"]
```

**O modelo calcula os números.** LangGraph organiza validação, atributos e explicação; a saída é estruturada e inspecionável. O texto explicativo padrão é determinístico e usa as contribuições reais do modelo. O VLM opcional observa o vídeo; não inventa métricas futuras.

Esse é o fluxo da aplicação de demonstração, que continua usando dados sintéticos. Os estudos reais de alcance e retenção são executados por scripts e relatórios próprios. O novo estudo de alcance preserva modelos e intervalos calibrados em `.runtime/reports/reach/`, mas não ativa o preditor da interface.

## O que está implementado

| Área | Entrega |
|---|---|
| Experiência visual | Visão geral, comparação, intervalos, explicações, avaliação, dados, treinamento e guia |
| Contratos de dados | Pydantic + JSON Schema; JSONL/JSON/Parquet; auditoria de origem, horizonte e disponibilidade temporal |
| Prevenção de vazamento | Histórico ponto a ponto no tempo; expurgo de rótulos imaturos; deduplicação; separação de campanhas |
| Modelagem | Ridge, CatBoost, LightGBM e blend; alvos absolutos/residuais; quantis calibrados por setor |
| Experimentos | Competição em três janelas, contexto forte, ablações, controles negativos, seeds e canais não vistos; seleção congelada e teste final de uso único |
| Testes públicos reais | Ganho modesto em 2016; transferência para 2020–2024 sem ganho; extração multimodal sem eficácia confirmada; resultados e previsões preservados |
| Vídeos completos | [Imagem, fala e edição](docs/FULL_VIDEO_METHOD.md) em 945 vídeos íntegros; teste reservado com 177 vídeos; vantagem não confirmada |
| Avaliação | MAE, MAE em log, MAPE com zeros explícitos, Spearman, pares, arrependimento de seleção, F1 de faixas, interval score, cobertura/largura e médias equilibradas por canal |
| Explicação | SHAP de árvores/decomposição linear, comparáveis históricos, evidências temporais, abstenção e rastros em JSON |
| Vídeo local | FFmpeg, atributos visuais/acústicos, cache por conteúdo e versão |
| Extratores opcionais | Whisper, OpenCLIP, OCR e rubrica estruturada via Ollama |
| Coleta | Snapshots oficiais de metadados do YouTube e montagem de alvos em horizonte fixo |
| Operação | Fila SQLite, retomada explícita de jobs interrompidos, registro de versões e promoção controlada |
| Integração | Função Python, FastAPI/OpenAPI e MCP stdio com saída estruturada |
| Validação prospectiva | Registro imutável antes da publicação; inclusão de resultado no horizonte; relatório de erro/cobertura |
| Qualidade | Testes de dados, modelo, API, vídeo real gerado e protocolo MCP; CI; lockfile |

## Navegue pela documentação

| Quero… | Leia |
|---|---|
| Saber se o modelo supera o histórico | [Teste externo mais recente](docs/TRANSFER_RESULTS.md) |
| Entender a seleção da base e do modelo | [Protocolo revisado](docs/REVISED_STUDY.md) |
| Saber se imagem, fala e edição ajudaram | [Resultado com vídeos completos](docs/FULL_VIDEO_RESULTS.md) |
| Conferir os testes anteriores de metadados | [Ken Jee e YouTube Trending](docs/REAL_DATA_RESULTS.md) |
| Avaliar a qualidade do modelo | [Case de modelagem e evidência](docs/MODEL_CASE.md) |
| Rodar e explorar | [Guia rápido](docs/QUICKSTART.md) |
| Entender módulos e fluxos | [Arquitetura](docs/ARCHITECTURE.md) |
| Importar uma base real sem vazamento | [Dados e coleta](docs/DATA.md) |
| Avaliar a validade dos resultados | [Metodologia](docs/METHODOLOGY.md) |
| Ativar extração multimodal | [Processamento de mídia](docs/MULTIMODAL.md) |
| Conectar a ferramenta à persona | [Python, API e MCP](docs/INTEGRATION.md) |
| Operar, monitorar e recuperar | [Operação](docs/OPERATIONS.md) |
| Apresentar o case com honestidade | [Roteiro de demonstração](docs/DEMO.md) |
| Ver decisões e limites de escopo | [Decisões técnicas](docs/DECISIONS.md) |
| Conferir o que foi testado | [Registro de verificação](docs/VERIFICATION.md) |

## Verifique a implementação

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest
uv run galaxies doctor
```

Os testes geram os próprios dados e não fazem chamadas a modelos remotos. O teste de vídeo usa FFmpeg quando disponível. Os extratores opcionais exigem instalação e pesos próprios; eles não são baixados pela suíte padrão.

## Estrutura

```text
src/galaxies/
  schemas.py       contratos e disponibilidade temporal
  data.py          auditoria, histórico e partições
  features.py      atributos e projeções ajustadas no treino
  media.py         vídeo, áudio, OCR, CLIP e VLM local
  estimators.py    famílias, alvos residuais e atribuições
  model.py         treino, calibração, avaliação e SHAP
  protocol.py      cortes, candidatos e regra de seleção
  research.py      competição, diagnósticos e avaliação final
  reach_study.py   alcance semanal, histórico, comparáveis e avaliação reservada
  public_benchmark.py  estudo público de metadados com alvos por calendário
  shortvideo.py      aquisição de vídeos completos e histórico de retenção
  deep_media.py     imagem, fala e semântica com pesos fixados
  video_structure.py  cortes em todos os frames e movimento
  retention_benchmark.py  teste temporal de retenção com escolha congelada
  experiments.py   suíte exploratória complementar
  service.py       fluxo LangGraph e decisões
  registry.py      versões, integridade e ativação
  jobs.py          fila persistente
  prospective.py   registro antes da publicação
  quality.py       monitoramento e concordância dos extratores
  api.py           HTTP e interface visual
  mcp_server.py    ferramentas para agentes
  cli.py           comandos reproduzíveis
  web/             interface sem dependências de frontend
docs/              documentação visual e contratos
examples/          requisições e configurações executáveis
tests/             verificações de comportamento
```

Dados, modelos, vídeos e rastros ficam em `.runtime/`, ignorado pelo Git. O repositório não contém arquivos do briefing, credenciais privadas, dados de clientes ou vídeos de terceiros. O acesso público ao dataset usa as informações de download publicadas pelos próprios autores.

Desenvolvido por **Guilherme Holanda**, no contexto Inteli Academy × Galaxies.
