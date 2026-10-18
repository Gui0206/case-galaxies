# Piloto de coleta — entrega de 20/10/2026

**Estado em 18/10/2026: INTERROMPIDO a pedido do usuário.** O LaunchAgent foi descarregado (`bootout`) e desabilitado (`disable`), inclusive para não voltar no próximo login. A automação anterior do Codex permanece pausada. Não há coleta ou aviso futuro ativo para este piloto. Não reativar sem nova autorização.

**Resultado até a interrupção:** consultas iniciais concluídas sem erro; zero vídeos novos registrados e zero resultados maduros. A última consulta bem-sucedida foi em 18/10/2026 às 10:50:03, horário de São Paulo. Isso não é evidência de ganho nem de funcionamento ao longo de vários dias.

O protocolo abaixo registra o que havia sido planejado, **não um acompanhamento em andamento**.

**Objetivo:** verificar se conseguimos registrar publicações novas e observar seu desempenho em uma janela consistente. Não treinar modelos nem apresentar a coleta como ganho preditivo.

| Decisão | Configuração |
|---|---|
| Canais | Doug DeMuro e carwow (automotivo); Alexandra Anele e NikkieTutorials (beleza) |
| Seleção | Conveniência, quatro canais públicos; não representa moda/beleza/automotivo como um todo |
| Entrada de novas publicações | 18 a 20/10, por 48 horas a partir da configuração |
| Acompanhamento | Até sete dias após o fechamento, com três horas de tolerância operacional |
| Frequência | A cada duas horas, por tarefa local do macOS |
| Dados | Identificador, canal, publicação, título inicial e suas versões observadas, duração, views, likes e horário exato da observação |
| Exclusões | Transmissões ao vivo não contam para maturidade; resultados ausentes não viram zero |
| Limites | 50 publicações recentes por canal por consulta; até 500 vídeos ativos; sem expansão automática |

A origem é a API oficial: [playlistItems.list](https://developers.google.com/youtube/v3/docs/playlistItems/list) descobre publicações e [videos.list](https://developers.google.com/youtube/v3/docs/videos/list) lê metadados e estatísticas. A credencial permanece no `.env` ignorado; não consta no agendamento ou nos logs.

## O que estava previsto para terça (cancelado)

Se o piloto tivesse continuado, seria possível produzir um registro reproduzível do funcionamento da coleta e das primeiras horas das publicações encontradas. Pode não haver publicações em alguns canais. Não haverá uma avaliação de views em sete dias para os vídeos capturados desde domingo. Os primeiros resultados maduros só podem surgir a partir de 25/10; os últimos, a partir de 27/10.

Não chamamos uma observação tardia de resultado exato aos sete dias: a janela aceitável é **168–171 horas**, com o horário efetivo preservado. Se o computador dormir/desligar e perder a janela, a lacuna é contada. Não há interpolação nem reconstrução de resultados passados.

## Execução e avisos

Antes da interrupção, o LaunchAgent `com.guilherme.galaxies-channel-pilot` estava configurado para executar `scripts/channel_pilot.py` a cada duas horas. Requer a sessão do usuário ativa e o Mac acordado e conectado; não impede o repouso. Há trava contra execuções sobrepostas. Após o encerramento, novas execuções não fazem consultas.

**Nenhuma chamada a LLM ocorre nas verificações.** A automação antiga do Codex continua pausada. Os avisos planejados eram notificações locais do macOS, sujeitos às permissões de notificação do sistema, somente ao encerrar as 48 horas, obter os primeiros resultados maduros, atingir três falhas consecutivas ou concluir o piloto. Não haverá mensagens periódicas neste chat.

- Estado e eventos: `.runtime/channel-pilot/status.json`.
- Resumo previsto pelo código: `.runtime/channel-pilot/summary.md`. Não chegou a ser gerado antes da interrupção; use o estado preservado e esta documentação.
- Dados e registro de execuções: `.runtime/channel-pilot/observations.sqlite`.
- Protocolo inicial versionado: `docs/results/channel-pilot/protocol.json`.
- Agendamento: `~/Library/LaunchAgents/com.guilherme.galaxies-channel-pilot.plist`.

Interrupção já executada: `launchctl bootout gui/$(id -u)/com.guilherme.galaxies-channel-pilot`, seguido de `launchctl disable gui/$(id -u)/com.guilherme.galaxies-channel-pilot`. Os arquivos permanecem locais. O `completed=false` no último estado significa que o protocolo não terminou; não significa que o agendamento continua ativo.

O código e o teste de janela temporal foram preservados como trabalho interrompido. Não foi realizada validação de operação prolongada, maturação, recuperação de falhas ou entrega efetiva de notificações. Nenhum teste foi reexecutado ao documentar a interrupção.

**Limites para o case:** esse piloto não contém vídeos completos, não recupera histórico anterior, não é uma amostra representativa e não comprova eficácia multimodal. Dá evidência de engenharia de dados; não justifica prometer uma grande elevação da nota ou uma previsão melhor. Nenhum treinamento está agendado.
