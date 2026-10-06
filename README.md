<p align="center"><img src="src/galaxies/web/mark.svg" width="64" alt="Galaxies Creative Intelligence"></p>

<h1 align="center">Galaxies · Creative Intelligence</h1>
<p align="center">Compare criativos antes da publicação.<br>Previsões estatísticas, incerteza explícita e evidências à vista.</p>

<p align="center"><a href="docs/QUICKSTART.md">Começar</a> · <a href="docs/METHODOLOGY.md">Entender a avaliação</a> · <a href="docs/DATA.md">Preparar dados</a> · <a href="docs/INTEGRATION.md">Integrar</a> · <a href="docs/DEMO.md">Apresentar o case</a></p>

> **Estado da evidência:** o sistema funciona de ponta a ponta com uma base sintética reproduzível. Nenhuma acurácia em campanhas reais é alegada. Uma base real autorizada e a validação prospectiva continuam necessárias para demonstrar utilidade de negócio.

![Visão geral do workspace](docs/assets/workbench.jpg)

## Execute em três comandos

Requisitos: [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.12 (gerenciado pelo uv) e FFmpeg para vídeos locais.

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
    F --> M["CatBoost<br>+ quantis calibrados"]
    M --> P["Previsão + intervalo<br>+ ranking"]
    P --> X["SHAP + comparáveis<br>+ limitações"]
    X --> T["Python · HTTP · MCP"]
```

**O modelo calcula os números.** LangGraph organiza validação, atributos e explicação; a saída é estruturada e inspecionável. O texto explicativo padrão é determinístico e usa as contribuições reais do modelo. O VLM opcional observa o vídeo; não inventa métricas futuras.

## O que está implementado

| Área | Entrega |
|---|---|
| Experiência visual | Visão geral, comparação, intervalos, explicações, avaliação, dados, treinamento e guia |
| Contratos de dados | Pydantic + JSON Schema; JSONL/JSON/Parquet; auditoria de origem, horizonte e disponibilidade temporal |
| Prevenção de vazamento | Histórico ponto a ponto no tempo; expurgo de rótulos imaturos; deduplicação; separação de campanhas |
| Modelagem | CatBoost, regressão quantílica e calibração conformal; baselines de média/mediana |
| Experimentos | Ablações, especialistas por setor, canais não vistos, Ridge e CatBoostRanker; teste final preservado na suíte |
| Avaliação | MAE, MAE em log, MAPE com zeros explícitos, Spearman, pares, arrependimento de seleção, F1 de faixas e cobertura |
| Explicação | SHAP nativo, comparáveis históricos, evidências temporais, abstenção e rastros em JSON |
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
  model.py         treino, calibração, avaliação e SHAP
  experiments.py   ablações e modelos comparativos
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

Dados, modelos, vídeos e rastros ficam em `.runtime/`, ignorado pelo Git. O repositório não contém arquivos do briefing, credenciais, dados de clientes ou vídeos de terceiros.

Desenvolvido por **Guilherme Holanda**, no contexto Inteli Academy × Galaxies.
