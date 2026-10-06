# O que foi verificado

Verificação local em 6 de outubro de 2026, Python 3.12.13, macOS. O [lockfile](../uv.lock) fixa as dependências. Este registro descreve uma execução; o resultado da CI remota precisa ser consultado separadamente.

**Resultado: 55 testes passaram**, sem skips no ambiente local com FFmpeg disponível. Ruff, formatação e build também passaram. A sintaxe JavaScript e a interface foram verificadas na entrega inicial; o reforço atual se concentrou na modelagem.

| Camada | Verificação realizada |
|---|---|
| Qualidade estática | Ruff e formatação; sintaxe do JavaScript com Node |
| Contratos e tempo | Rejeição de vazamento, horizonte incorreto, escalas inválidas, mistura sintética/real e timestamps futuros |
| Dados | JSON/JSONL/Parquet, deduplicação, campanhas, canais excluídos e histórico disponível na decisão |
| Modelo | Treinamento real em dados sintéticos; persistência; identidade aditiva do SHAP; quantil conformal; detecção de alteração dos artefatos |
| Competição | Três janelas com avaliações disjuntas; mesmos IDs; isolamento do teste final mesmo ao alterar seus alvos; rejeição de candidatos incompletos |
| Seleção e confirmação | Regra protege erro de contagens; hashes rejeitam alteração de dados/código; avaliação final de uso único; promoção real exige seleção verificada |
| Famílias | Ridge, CatBoost, LightGBM e blend em alvos absolutos e residuais; persistência nativa e atribuições aditivas |
| Diagnósticos | Embaralhamento preserva contexto/alvos; interval score penaliza amplitude e falhas |
| Decisão | Modelo sintético e entrada simulada bloqueiam recomendação; atributos ausentes são sinalizados |
| API | Contratos, rastros, autenticação, restrição de origem e importação inválida |
| Treinamento assíncrono | Job persistido chega a `completed`; processo MCP/API vivo não é interrompido por outra instância |
| MCP | Cliente oficial inicia servidor stdio, lista ferramentas e recebe `structuredContent` |
| Mídia | FFmpeg gera um vídeo de teste original; o extrator lê frames/áudio e reutiliza o cache |
| Prospectivo | Resultado no horizonte correto; rejeição de sobrescrita e de evidência sintética |
| Monitoramento | Ausência diferente de zero; latência; separação de níveis de evidência; concordância com anotação humana |
| Interface | Navegação, comparação, evidências, troca de versão, upload/extração de vídeo, treinamento e layouts de desktop/tela estreita |
| Distribuição | Build de wheel e source archive; HTML, CSS, JavaScript e SVG incluídos no pacote |

## Reproduzir

```bash
uv sync --frozen --group dev
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest --cov=galaxies --cov-report=term-missing
node --check src/galaxies/web/app.js
uv build
```

Os testes usam diretórios temporários, dados sintéticos determinísticos e nenhuma credencial externa. O processo MCP é separado; a cobertura padrão do processo pai não atribui suas linhas, embora o teste de protocolo o execute. A suíte de experimentos e os comandos CLI também foram executados manualmente. A competição completa de 12 configurações, ablações, controles negativos, seeds e canais excluídos tem [resultados salvos](results/research/development.json); a [avaliação final](results/research/final.json) confirma uma seleção previamente congelada em dados sintéticos.

A cobertura instrumentada foi 73% no total, 99% em `estimators`, 97% em `metrics`, 91% em `protocol` e 91% em `research`. CLI, MCP subprocessado e extratores opcionais limitam essa medida; cobertura de linhas não equivale a validade estatística. A comparação HTTP com o modelo residual selecionado também foi confrontada com a saída CLI.

## Ainda exige ambiente e dados externos

- Desempenho preditivo em publicações/campanhas reais.
- Qualidade de Whisper, OpenCLIP, OCR e VLM, incluindo avaliação cega da rubrica.
- Coleta autenticada de snapshots do YouTube.
- Conexão com a plataforma interna da Galaxies.
- Operação multiusuário, carga e identidade corporativa.

Essas verificações não são substituídas por exemplos artificiais, mocks ou alteração de rótulos.
