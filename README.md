<p align="center"><img src="src/galaxies/web/mark.svg" width="64" alt="Galaxies Creative Intelligence"></p>

<h1 align="center">Galaxies · Creative Intelligence</h1>
<p align="center">Imagem, fala e edição acrescentam valor ao histórico do canal?<br>Análise de vídeos e avaliação temporal com dados reais.</p>

<p align="center"><a href="docs/FULL_VIDEO_RESULTS.md">Resultado com vídeos completos</a> · <a href="docs/FULL_VIDEO_METHOD.md">Como a análise funciona</a> · <a href="docs/MODEL_CASE.md">O case de modelagem</a> · <a href="docs/QUICKSTART.md">Começar</a> · <a href="docs/DATA.md">Preparar dados</a></p>

> **Resultado real — 17/10/2026:** a análise de imagem, fala e edição foi implementada e testada em vídeos reservados. **Ainda não demonstrou vantagem confiável sobre o histórico.** O estudo mede retenção no painel de observação; não valida visualizações/curtidas após publicação nem campanhas da Galaxies. Nenhum modelo foi promovido para produção.

## A resposta em números

Baixamos 954 arquivos do Tsinghua ShortVideo e aceitamos **945 vídeos íntegros**, após nove exclusões documentadas. Executamos análise de imagem, transcrição e edição em 13,8 horas de vídeo. A escolha dos modelos usou períodos anteriores; o teste final reservou **177 vídeos de 164 criadores**.

| Como prever a retenção | Erro médio no teste ↓ |
|---|---:|
| Histórico do criador ajustado pela categoria | 24,1 pontos percentuais |
| Histórico, categoria e duração | 23,3 pontos percentuais |
| Contexto, título, imagem, fala/som e edição | 23,7 pontos percentuais |

Retenção é quanto do vídeo as pessoas assistem. Prever 60% quando o observado é 50% representa erro de dez pontos percentuais. A pequena melhora de **1,6% no erro** sobre o histórico ajustado é incerta; a análise completa não superou a referência de contexto no erro observado. Apenas 27,7% das previsões completas ficaram a até dez pontos do valor real.

![Resultado com vídeos completos](docs/assets/full-video-results.png)

**Leia primeiro o [resultado explicado](docs/FULL_VIDEO_RESULTS.md)**: comparações, incerteza, exemplos que incluem erros e arquivos auditáveis. O [método](docs/FULL_VIDEO_METHOD.md) mostra os atributos extraídos e os comandos para analisar um vídeo seu. Não se atribui uma “nota de viralidade” sem validação.

Os [estudos anteriores de metadados](docs/REAL_DATA_RESULTS.md), com Ken Jee e YouTube Trending, também não demonstraram ganho consistente. Seus testes com 30 e 377 vídeos permanecem preservados; não foram reutilizados para ajustar este estudo.

## A qualidade está na avaliação do modelo

O núcleo compara **12 configurações de Ridge, CatBoost, LightGBM e blend em três janelas temporais**, congela a escolha e só então abre uma avaliação final. O conteúdo precisa acrescentar valor sobre uma referência forte que já conhece o canal. Contagens, ranking, incerteza, ablações e canais novos são avaliados separadamente.

O **[case de modelagem](docs/MODEL_CASE.md)** descreve a arquitetura e o estudo sintético anterior. Os estudos públicos usam duas janelas de desenvolvimento devido ao tamanho das amostras após expurgo temporal. O estudo de retenção compara cinco configurações em seis grupos de atributos, preserva a escolha congelada e registra as exclusões. A evidência sintética verifica a implementação, mas não substitui o resultado real acima.

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

Esse é o fluxo da aplicação de demonstração, que continua usando dados sintéticos. O estudo real de retenção é executado pelos scripts e relatórios próprios; não calibra intervalos individuais nem ativa o preditor de contagens da interface.

## O que está implementado

| Área | Entrega |
|---|---|
| Experiência visual | Visão geral, comparação, intervalos, explicações, avaliação, dados, treinamento e guia |
| Contratos de dados | Pydantic + JSON Schema; JSONL/JSON/Parquet; auditoria de origem, horizonte e disponibilidade temporal |
| Prevenção de vazamento | Histórico ponto a ponto no tempo; expurgo de rótulos imaturos; deduplicação; separação de campanhas |
| Modelagem | Ridge, CatBoost, LightGBM e blend; alvos absolutos/residuais; quantis calibrados por setor |
| Experimentos | Competição em três janelas, contexto forte, ablações, controles negativos, seeds e canais não vistos; seleção congelada e teste final de uso único |
| Testes públicos reais | Dois estudos de metadados, duas janelas de desenvolvimento, versões/hashes das fontes, referências simples e resultados negativos preservados |
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
