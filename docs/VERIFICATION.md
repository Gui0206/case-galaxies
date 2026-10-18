# O que foi verificado

## Estado mais recente documentado — 18/10/2026

Esta página separa **verificação de software**, **resultado preditivo real** e **partes não validadas**. Esta revisão foi somente documental: não executou suíte, inferência, coleta ou avaliação. Os totais de 81 e 88 testes nas seções antigas são registros de suas respectivas datas, não a contagem atual nem uma afirmação de que toda a suíte passou agora.

### Avaliação real SMTPD encerrada

O [relatório](SMTPD_RESULTS.md), as [515 previsões individuais](results/smtpd/final/test_predictions.csv) e a [verificação independente já salva](results/smtpd/final/verification.json) registram o teste de 165 canais. Erro macro em log: histórico **0,686802**, contexto **0,685355**, conteúdo **0,685569**. A redução de erro de 0,18% sobre o histórico não confirmou vantagem; contra contexto houve piora de 0,03%.

O recorte multimodal contém 219 vídeos completos do teste. Imagem, fala e edição foram realmente executadas e comparadas; a combinação completa não melhorou a referência textual. A [auditoria de mídia](results/smtpd/final/media-audit.json) contabiliza 669 itens previstos, 667 baixados e 662 extrações válidas. Falhas não foram substituídas por atributos inventados.

### Integração já verificada antes desta revisão

- O adaptador de inferência reproduziu seis previsões publicadas com e sem mídia, incluindo as referências e o ordenador. Isso verifica carregamento e cálculo; não constitui novo teste de generalização.
- Alterar resultados do próprio vídeo ou de publicações futuras não mudou a previsão verificada.
- Python, HTTP e MCP foram conectados aos artefatos reais. Os testes de integração local usam os artefatos existentes e bloqueiam rede e ajuste nas verificações de inferência; podem ser omitidos em ambientes sem esses arquivos locais.
- A interface foi verificada nos dois caminhos: vídeo próprio descritivo e exemplo histórico com previsão. A integração descritiva foi testada bloqueando a chamada ao preditor de views.
- Esses registros não equivalem a uma nova execução da suíte completa, a validação de produção ou a um novo resultado estatístico.

### Ainda não verificado

Ganho prospectivo em campanhas atuais; previsão para novos canais/clientes; precisão dos extratores por anotação humana cega; rubrica VLM e OCR como fontes de ganho; operação prolongada do piloto e entrega efetiva das notificações locais. A rubrica VLM não foi validada apenas por existir um adaptador. O piloto foi interrompido com zero vídeos registrados e não adicionou evidência de eficácia. [Estado final e pendências](CURRENT_STATE.md).

## Registro histórico de software — 17/10/2026

Registro histórico de verificação local em **17 de outubro de 2026**, Python 3.12.13, macOS. O [lockfile](../uv.lock) fixa as dependências. Este registro descreve execuções locais; o resultado da CI remota precisa ser consultado separadamente.

**Resultado daquela execução: 81 testes passaram**, sem skips no ambiente local com FFmpeg e os extras do estudo multimodal instalados. Ruff, formatação e build também passaram. Houve sete avisos de depreciação de conversão temporal pandas/NumPy no estudo anterior, sem falhas. A sintaxe JavaScript e a interface foram verificadas na entrega inicial; esta atualização acrescenta o estudo de alcance à análise de vídeos completos e ao teste temporal de retenção.

A [CI do congelamento do novo estudo `660999e`](https://github.com/Gui0206/case-galaxies/actions/runs/39080897379) também terminou com sucesso, antes da publicação dos resultados finais. A CI usa o ambiente básico Linux; ela não baixa pesos nem reproduz a extração MLX executada localmente.

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
| Alcance semanal | Sete dias explícitos; resultados imaturos e canais reservados não entram no histórico; vocabulário e projeções ajustados só no treino; persistência de árvores/Ridge/encoder; seleção congelada e abertura final única |
| Retenção | Histórico aguarda o fim da observação; expurgo temporal; PCA somente no treino; remoção efetiva de modalidades; bootstrap pareado; escolha congelada e teste final de uso único |
| Prospectivo | Resultado no horizonte correto; rejeição de sobrescrita e de evidência sintética |
| Monitoramento | Ausência diferente de zero; latência; separação de níveis de evidência; concordância com anotação humana |
| Interface | Navegação, comparação, evidências, troca de versão, upload/extração de vídeo, treinamento e layouts de desktop/tela estreita |
| Distribuição | Build de wheel e source archive; HTML, CSS, JavaScript e SVG incluídos no pacote |

## Comandos de referência para verificações futuras (não executados nesta revisão)

```bash
uv sync --frozen --group dev
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest --cov=galaxies --cov-report=term-missing
node --check src/galaxies/web/app.js
uv build
```

Os testes do núcleo histórico usam diretórios temporários, dados sintéticos determinísticos e nenhuma credencial externa. Os testes posteriores de integração SMTPD também dependem de artefatos reais locais, quando presentes. O processo MCP é separado; a cobertura padrão do processo pai não atribui suas linhas, embora o teste de protocolo o execute. A suíte de experimentos e os comandos CLI também foram executados manualmente. A competição completa de 12 configurações, ablações, controles negativos, seeds e canais excluídos tem [resultados salvos](results/research/development.json); a [avaliação final](results/research/final.json) confirma uma seleção previamente congelada em dados sintéticos.

A cobertura instrumentada **da execução anterior, em 06/10**, foi 73% no total, 99% em `estimators`, 97% em `metrics`, 91% em `protocol` e 91% em `research`; não foi recalculada para esta atualização. CLI, MCP subprocessado e extratores opcionais limitam essa medida; cobertura de linhas não equivale a validade estatística. A comparação HTTP com o modelo residual selecionado também foi confrontada com a saída CLI na entrega anterior.

## Evidência empírica adicionada em 17/10

O [estudo real](REAL_DATA_RESULTS.md) treinou 12 configurações em duas janelas de desenvolvimento para cada uma de duas bases públicas. Seleção, dados e código foram congelados antes dos testes finais com 30 vídeos do Ken Jee e 377 do Trending. O CatBoost residual anterior, a seleção automática e uma referência aprendida de contexto foram comparados com medianas históricas. Modelos e previsões estão preservados em `.runtime/reports/public/`; os relatórios auditáveis estão em [docs/results/public](results/public/).

**Resultado: nenhum ganho consistente sobre o histórico foi demonstrado.** São dados reais, mas apenas metadados retrospectivos, com alvos por calendário e limitações de seleção e disponibilidade. O estudo não certifica vídeo/áudio completos, publicação prospectiva ou campanhas da Galaxies. O campo `eligible_for_production` permanece `false` nos dois relatórios.

## Ainda exige ambiente e dados externos

O estudo de vídeos completos descrito a seguir acrescenta execução real de Whisper/OpenCLIP e edição, mas não uma medição humana da precisão desses extratores.

- Ganho preditivo consistente em campanhas representativas da Galaxies e validação prospectiva; a melhora no recorte público de 2016 não valida essa transferência.
- Qualidade de Whisper, OpenCLIP, OCR e VLM, incluindo avaliação cega da rubrica.
- Acompanhamento contínuo de uma coorte prospectiva até a maturidade. A API oficial já foi usada para metadados SMTPD e consultas iniciais do piloto; não confundir isso com coleta contínua validada.
- Conexão com a plataforma interna da Galaxies.
- Operação multiusuário, carga e identidade corporativa.

Essas verificações não são substituídas por exemplos artificiais, mocks ou alteração de rótulos.

## Vídeos completos: execução e resultado

O [estudo de retenção](FULL_VIDEO_RESULTS.md) baixou 954 arquivos (12,59 GB) e aceitou 945: oito divergiam da duração dos logs e um estava truncado na própria fonte. A segunda aquisição desse arquivo produziu o mesmo hash. A validação bloqueou o treinamento até a investigação e o registro da exclusão; nenhuma falha foi substituída por atributos inventados.

Foram analisadas 13,81 horas: 1.984.272 frames decodificados, 14.173 imagens enviadas ao encoder visual e 19.761 trechos de fala, em 869 vídeos com transcrição. Os outros 76 não tiveram fala detectada/transcrita; transcrição vazia é diferente de falha de processamento. As verificações de duração, tempos, dimensões e valores finitos passaram nos 945 registros aceitos.

O desenvolvimento executou cinco configurações em seis grupos de atributos e duas janelas temporais. A seleção foi congelada antes do teste final com 177 vídeos de 164 criadores, usando 623 vídeos no treino final. Erro médio: histórico ajustado **24,10 pp**, contexto **23,29 pp**, análise completa **23,70 pp**. O ganho da análise completa sobre o histórico ajustado foi **0,40 pp**, com intervalo de 95% de **−1,41 a 2,11 pp**. O critério de sucesso não foi cumprido e `eligible_for_production` permanece `false`.

As métricas foram recalculadas independentemente a partir das previsões individuais e coincidiram com o relatório. A análise avulsa via CLI foi executada em arquivo real e preservou a identidade do extrator do benchmark, usando cache separado. Os gráficos foram renderizados e inspecionados. Relatórios e previsões estão em [results/multimodal](results/multimodal/); vídeos, transcrições completas e modelos treinados permanecem em `.runtime/`.

## Alcance semanal: avaliação final nova

O [novo estudo](REACH_RESULTS.md) preparou 263.317 vídeos de 68.411 canais do Beyond Views, categorias Automotivo e Howto & Style. A aquisição validou o MD5 integral publicado pela fonte e o SHA-256 de cada membro extraído. Foram exigidas sete observações diárias explícitas; resultados históricos entram somente após nove datas.

O desenvolvimento passou por uma revisão da referência residual e uma correção declarada da regra de seleção. As rodadas anteriores estão preservadas, sem consulta aos resultados finais. As duas execuções da rodada revisada produziram **o mesmo hash de previsões de desenvolvimento**, demonstrando que a última alteração afetou a escolha, não os ajustes. O commit `660999e` publicou a configuração congelada no GitHub antes da abertura final.

O teste principal contém **27.505 vídeos de 10.621 canais**, com 34.598 pares em 4.243 canais. O candidato selecionado reduziu o MAE macro em log de **0,550862 para 0,530242**, melhora de **3,743%**, com IC95% da redução absoluta **[0,018037; 0,023243]**. O acerto de ordenação foi **60,63%**, contra **49,31%** do histórico e **58,88%** do contexto. O ganho de ordenação sobre contexto foi **1,75 pp**, IC95% **[0,75; 2,78]**. A hipótese de ordenação cumpriu o critério registrado; a de redução relevante do erro (5% contra histórico e contexto) não cumpriu. `case_solved` e `production_eligible` permanecem `false`.

A verificação independente recalculou erro em log, erro em visualizações, cobertura, proporção dentro de um fator dois e quantidade de resultados históricos maduros. Uma implementação que enumera pares diretamente reproduziu o ranking e seus intervalos, conferindo também a amostragem por canal. Todas as verificações passaram. Os modelos salvos reproduziram as previsões de calibração antes da publicação dos resultados. O gráfico foi renderizado e inspecionado.

Os dados têm limitações substantivas: amostra selecionada de 2016, metadados retrospectivos, ausência dos arquivos originais e de informação de mídia paga. O teste não mede ganho de imagem/fala/edição sobre visualizações, não valida curtidas e não comprova benefício em campanhas atuais. Todos os resultados, inclusive recortes desfavoráveis e controles não escolhidos, estão em [results/reach](results/reach/).


## Transferência para 2020–2024

Os 88 testes passaram antes da abertura externa. A configuração foi publicada em `0138217` antes de executar o teste final. As métricas de 922 vídeos/119 canais foram recalculadas independentemente a partir das previsões: histórico 0,424454, contexto 0,419404, conteúdo 0,438628 de erro macro em log. Resultado desfavorável, sem promoção para produção. [Relatório e limitações](TRANSFER_RESULTS.md).
