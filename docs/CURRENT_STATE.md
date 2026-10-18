# Estado da entrega — 18/10/2026

**O usuário pediu para parar o trabalho e depois apenas garantir a documentação. Não retomar desenvolvimento, coletas, treinos, avaliações ou automações sem nova solicitação.** A entrega do case foi informada para terça-feira, 20/10/2026.

## Entregue e demonstrado

- Modelos reais SMTPD congelados conectados a Python, API e MCP; guia em [REAL_DEMO.md](REAL_DEMO.md).
- Interface local em `http://127.0.0.1:8000/`, disponível enquanto o servidor estiver ativo. Não há hospedagem pública ou inicialização automática do servidor documentada como entrega.
- **Meu vídeo:** upload local e descrição de fala, edição e medidas visuais. Não pede canal nem calcula views, nota criativa ou probabilidade de viralização.
- **Canais da base:** três exemplos publicados, sem seleção pelo acerto, com previsão, histórico, intervalo e resultado observado anexado depois da inferência.
- Demo sintética antiga isolada em `/synthetic`.
- Os dois fluxos foram verificados no navegador; a integração da análise própria foi testada bloqueando a chamada ao preditor. Não houve novo treino para essas mudanças.

Commits principais: `5122887` (adaptador real), `5be16b7` (interface real), `3e308c2` (separação entre descrição própria e previsão histórica), publicados no GitHub com a identidade Git de Guilherme Holanda.

## Evidência e limite principal

A avaliação SMTPD já foi encerrada e verificada. Em 515 vídeos de 165 canais, o erro do modelo foi 0,18% menor que o histórico e 0,03% maior que contexto, sem ganho estatisticamente confirmado. Em 219 vídeos completos, a combinação de imagem, fala e edição teve erro cerca de 0,5% maior que a referência textual. Veja [SMTPD_RESULTS.md](SMTPD_RESULTS.md) e os artefatos em `docs/results/smtpd/final/`.

Não há previsão validada para o canal de um novo usuário. Um canal da base não substitui o histórico próprio. A análise descritiva é um complemento funcional, não uma solução para a falta de ganho preditivo. Não existe canal parceiro disponível. A nota de 6/10 discutida no chat é uma avaliação subjetiva da resolução, não uma métrica do modelo.

Não apagar `evaluation-opened.json`, reabrir o teste ou ajustar modelos pelos resultados reservados. Melhor apresentação e infraestrutura não mudam a conclusão estatística.

## Piloto iniciado e interrompido

Foi preparado um piloto de metadados públicos para quatro canais: Doug DeMuro, carwow, Alexandra Anele e NikkieTutorials. O plano foi limitado a 48 horas de entrada de novas publicações, com acompanhamento até sete dias depois e tolerância operacional de três horas. A janela de 48 horas não permitiria comprovar ganho no alvo de sete dias antes da entrega.

A API respondeu às consultas iniciais; o último estado tinha **zero vídeos novos e zero resultados maduros**, sem erro. O usuário então mandou parar. O LaunchAgent foi descarregado e desabilitado; a automação antiga `concluir-valida-o-galaxies` continua pausada. **Não há acompanhamento ativo do piloto nem promessa de aviso futuro.**

O script e seu teste foram preservados, sem alegação de validação prolongada. Protocolo, caminhos locais, lacunas e registro da interrupção: [CHANNEL_PILOT.md](CHANNEL_PILOT.md). Não foram coletados vídeos completos nesse piloto, treinados modelos ou feitos novos testes preditivos.

## O que permanece local

Pesos, dados, vídeos, caches, banco SQLite e credenciais não são distribuídos pelo Git. Os manifestos e documentos versionados identificam os artefatos. A chave permanece no `.env` ignorado e não foi copiada para a documentação.

O último estado bruto do coletor permanece em `.runtime/channel-pilot/status.json`; o estado operacional de interrupção está registrado em `docs/results/channel-pilot/stopped-state.json`. Não confundir `completed=false` do coletor com um agendamento ativo.

## Pendências explícitas, sem execução autorizada

- Ganho preditivo relevante e generalizável continua não demonstrado.
- Não foi construído histórico próprio de um usuário nem uma base representativa de campanhas atuais.
- O piloto não atingiu maturidade; não há resultados a avaliar ou apresentar como melhora do modelo.
- Não foi validada a operação do piloto por vários dias nem a entrega efetiva das notificações locais.
- Não há novo treino, coleta ou busca de dados aprovado após a ordem de interrupção.

A atualização documental final não executou modelos, testes, coleta ou notificações.
