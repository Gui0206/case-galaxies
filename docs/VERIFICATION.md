# O que foi verificado

Última verificação local em **17 de outubro de 2026**, Python 3.12.13, macOS. O [lockfile](../uv.lock) fixa as dependências. Este registro descreve execuções locais; o resultado da CI remota precisa ser consultado separadamente.

**Resultado atual: 73 testes passaram**, sem skips no ambiente local com FFmpeg e os extras do estudo multimodal instalados. Ruff, formatação e build também passaram. Houve sete avisos de depreciação de conversão temporal pandas/NumPy no estudo anterior, sem falhas. A sintaxe JavaScript e a interface foram verificadas na entrega inicial; esta atualização se concentra na análise de vídeos completos e no teste temporal de retenção.

A [CI do commit de validação da extração `050d0da`](https://github.com/Gui0206/case-galaxies/actions/runs/39076952015) também terminou com sucesso. A CI usa o ambiente básico Linux; ela não baixa pesos nem reproduz a extração MLX executada localmente.

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
| Bases públicas | Sete testes novos: dias ausentes não viram zero; snapshots no horizonte; deduplicação entre regiões e por canal/título; histórico sem resultados futuros; encoder ajustado só no treino; hashes e teste final único; reamostragem pareada |
| Decisão | Modelo sintético e entrada simulada bloqueiam recomendação; atributos ausentes são sinalizados |
| API | Contratos, rastros, autenticação, restrição de origem e importação inválida |
| Treinamento assíncrono | Job persistido chega a `completed`; processo MCP/API vivo não é interrompido por outra instância |
| MCP | Cliente oficial inicia servidor stdio, lista ferramentas e recebe `structuredContent` |
| Mídia | FFmpeg gera um vídeo de teste original; o extrator lê frames/áudio e reutiliza o cache |
| Vídeo completo | Corte conhecido em vídeo original de duas cenas; timestamps de fala recolocados após remoção de silêncio; rejeição de decodificação incompleta e vetores inválidos |
| Retenção | Histórico aguarda o fim da observação; expurgo temporal; PCA somente no treino; remoção efetiva de modalidades; bootstrap pareado; escolha congelada e teste final de uso único |
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

A cobertura instrumentada **da execução anterior, em 06/10**, foi 73% no total, 99% em `estimators`, 97% em `metrics`, 91% em `protocol` e 91% em `research`; não foi recalculada para esta atualização. CLI, MCP subprocessado e extratores opcionais limitam essa medida; cobertura de linhas não equivale a validade estatística. A comparação HTTP com o modelo residual selecionado também foi confrontada com a saída CLI na entrega anterior.

## Evidência empírica adicionada em 17/10

O [estudo real](REAL_DATA_RESULTS.md) treinou 12 configurações em duas janelas de desenvolvimento para cada uma de duas bases públicas. Seleção, dados e código foram congelados antes dos testes finais com 30 vídeos do Ken Jee e 377 do Trending. O CatBoost residual anterior, a seleção automática e uma referência aprendida de contexto foram comparados com medianas históricas. Modelos e previsões estão preservados em `.runtime/reports/public/`; os relatórios auditáveis estão em [docs/results/public](results/public/).

**Resultado: nenhum ganho consistente sobre o histórico foi demonstrado.** São dados reais, mas apenas metadados retrospectivos, com alvos por calendário e limitações de seleção e disponibilidade. O estudo não certifica vídeo/áudio completos, publicação prospectiva ou campanhas da Galaxies. O campo `eligible_for_production` permanece `false` nos dois relatórios.

## Ainda exige ambiente e dados externos

O estudo de vídeos completos descrito a seguir acrescenta execução real de Whisper/OpenCLIP e edição, mas não uma medição humana da precisão desses extratores.

- Ganho preditivo consistente em campanhas representativas da Galaxies e validação prospectiva; o teste público de metadados já executado não demonstrou esse ganho.
- Qualidade de Whisper, OpenCLIP, OCR e VLM, incluindo avaliação cega da rubrica.
- Coleta autenticada de snapshots do YouTube.
- Conexão com a plataforma interna da Galaxies.
- Operação multiusuário, carga e identidade corporativa.

Essas verificações não são substituídas por exemplos artificiais, mocks ou alteração de rótulos.

## Vídeos completos: execução e resultado

O [estudo de retenção](FULL_VIDEO_RESULTS.md) baixou 954 arquivos (12,59 GB) e aceitou 945: oito divergiam da duração dos logs e um estava truncado na própria fonte. A segunda aquisição desse arquivo produziu o mesmo hash. A validação bloqueou o treinamento até a investigação e o registro da exclusão; nenhuma falha foi substituída por atributos inventados.

Foram analisadas 13,81 horas: 1.984.272 frames decodificados, 14.173 imagens enviadas ao encoder visual e 19.761 trechos de fala, em 869 vídeos com transcrição. Os outros 76 não tiveram fala detectada/transcrita; transcrição vazia é diferente de falha de processamento. As verificações de duração, tempos, dimensões e valores finitos passaram nos 945 registros aceitos.

O desenvolvimento executou cinco configurações em seis grupos de atributos e duas janelas temporais. A seleção foi congelada antes do teste final com 177 vídeos de 164 criadores, usando 623 vídeos no treino final. Erro médio: histórico ajustado **24,10 pp**, contexto **23,29 pp**, análise completa **23,70 pp**. O ganho da análise completa sobre o histórico ajustado foi **0,40 pp**, com intervalo de 95% de **−1,41 a 2,11 pp**. O critério de sucesso não foi cumprido e `eligible_for_production` permanece `false`.

As métricas foram recalculadas independentemente a partir das previsões individuais e coincidiram com o relatório. A análise avulsa via CLI foi executada em arquivo real e preservou a identidade do extrator do benchmark, usando cache separado. Os gráficos foram renderizados e inspecionados. Relatórios e previsões estão em [results/multimodal](results/multimodal/); vídeos, transcrições completas e modelos treinados permanecem em `.runtime/`.
