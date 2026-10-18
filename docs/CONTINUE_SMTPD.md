# Retomar sem repetir trabalho

**Etapa concluída:** o teste reservado foi aberto uma vez, verificado e documentado em [SMTPD_RESULTS.md](SMTPD_RESULTS.md). Não há ganho confirmado. Reutilize as previsões publicadas para conferir os números; não repita coleta ou treinamento.

Comece com **um comando de leitura**, sem baixar dados nem carregar modelos:

```bash
uv run python scripts/smtpd_status.py
```

Antes de iniciar outro processo, confira os existentes:

```bash
ps -axo pid,etime,command | rg 'collect_smtpd_media|extract_smtpd_media|develop_smtpd_extensions|finalize_smtpd|smtpd_finish_worker'
```

## Caminho restante

| Etapa | Sinal de conclusão | Próxima ação |
|---|---|---|
| Coleta declarada: 669 vídeos | `media-acquisition.json`: `complete=true` | Preservar falhas; não baixar tudo novamente |
| Extração de imagem, fala e edição | `features/index.json`: `extraction_complete=true`, coleta completa | Reutilizar os atributos com hash válido |
| Escolha multimodal | `smtpd-media/selection.json` existente | Ler o resultado; não repetir o treino |
| Congelamento | `smtpd-final/selection.json` existente | Publicar seleção e hashes **antes** do teste |
| Teste reservado | `smtpd-final/final.json` existente | Recalcular métricas, sem retreinar |
| Verificação | `smtpd-final/verification.json` existente | Gerar figura/relatório e atualizar README |

Arquivos de coleta ficam em `.runtime/public/smtpd/`; relatórios de pesquisa em `.runtime/reports/`. Logs ativos: `media-acquisition.log`, `media-extraction.log` e `media-development.log`, dentro da pasta de dados.

Uma continuação local em `.runtime/smtpd_finish_worker.py` aguarda a mídia e executa congelamento, commit/push da seleção, teste único, verificação e geração do relatório. O status informa sua fase e se o processo está vivo. **Enquanto estiver vivo, não execute essas etapas em paralelo.** O log fica em `.runtime/public/smtpd/completion-worker.log`. Depois de pronto, ainda é necessário inspecionar a figura, atualizar o README e publicar o resultado. Se o worker parar, sua causa fica em `completion-worker.json`; preservar esse registro antes de uma retomada revisada.

Os comandos finais, **somente após a etapa anterior estar concluída**, são:

```bash
uv run python scripts/finalize_smtpd.py freeze
# Copiar seleção, desenvolvimento e proveniência para docs/results/smtpd/final/.
# Fazer commit e push antes do comando seguinte.
uv run python scripts/finalize_smtpd.py final
uv run python scripts/verify_smtpd_results.py
uv run python scripts/render_smtpd_results.py
```

Se `evaluation-opened.json` já existe, **não executar `final` de novo nem apagar o registro**. Inspecionar o resultado ou uma eventual falha técnica, preservando seu histórico. A verificação e a renderização podem ser repetidas: não escolhem modelos e não abrem um novo teste.

## Decisões já encerradas

- Base principal: SMTPD, views nominais aos sete dias, publicação/canal recuperados oficialmente. Teste: 515 vídeos de 165 canais, a partir de 05/01/2024.
- Conteúdo textual: combinação de contexto e regressão semântica dentro do canal; erro de desenvolvimento 0,6335.
- Mais dimensões de texto e regressão lexical esparsa não melhoraram a escolha.
- Transferência de 40.781 vídeos de outros setores não ajudou. Seu controle sem transferência fortaleceu a referência de contexto: 0,6357.
- Ponderar a transferência pela proximidade dos setores não atingiu o critério de adoção.
- Treinar diretamente para ordenação obteve 58,5%, abaixo dos 63,8% do modelo já existente. Não acrescentar essa família à confirmação.
- Mídia concluída: 669 itens contabilizados, 667 downloads e 662 extrações válidas. Duas falhas de acesso e cinco de extração foram preservadas; a avaliação principal manteve todos os 515 vídeos, usando texto quando faltava mídia.
- Os estudos anteriores de alcance, transferência, adaptação recente e retenção já abriram seus testes. Seus resultados estão publicados; não reutilizá-los para escolher uma nova tentativa.

## Regra para gastar tempo

Uma nova tentativa precisa de **hipótese diferente, resultado que possa mudar a decisão e critério de encerramento antes de rodar**. Na etapa atual, concluir a avaliação reservada tem prioridade sobre ampliar grids, refazer buscas de bases, adicionar interface ou repetir experimentos negativos. Melhorar documentação não exige repetir treinamento.

A coleta retoma apenas itens pendentes. A extração usa cache por conteúdo e versão. Bloqueios ficam registrados e exigem revisão da causa; não reiniciar processos repetidamente. Credenciais ficam exclusivamente no `.env` local ignorado pelo Git — nunca imprimir seu conteúdo.
