# Próxima evidência: melhorar a decisão, sem reciclar o teste

Estado inicial: 18/10/2026 UTC. Os testes de alcance de 2016, retenção e metadados anteriores estão encerrados. A nota de trabalho é 6,5/10; não será elevada por quantidade de código, tamanho de encoder ou número de tentativas.

## Pergunta que orienta esta rodada

Conseguimos escolher melhor entre vídeos do mesmo canal e prever alcance antes da publicação, em dados diferentes dos estudos anteriores? Uma vitória em dados selecionados retrospectivamente continuará sendo limitada àquela população.

Ordem de investimento: (1) validar fonte, datas e alvo; (2) testar aprendizagem direta de ordenação contra referências fortes; (3) acrescentar modalidades somente onde existirem vídeos e alvos adequados; (4) confirmar uma configuração congelada em dados novos. Um caminho é interrompido se sua limitação impede responder à pergunta, mesmo que prometa métricas altas.

## Auditoria nova de fontes

| Fonte primária | Situação antes de modelar |
|---|---|
| [YouTube Trending 2020–2024, Rishav Sharma](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset) | API pública acessível; versão 1346, de 15/04/2024, CC0 declarada. Tem snapshots, publicação e canal. Seleção por entrar em tendências: não representa todos os uploads. Auditar tamanho e disponibilidade; não confundir com o teste anterior de 2017–2018. |
| [AMPS, autores do artigo](https://github.com/dxlabskku/AMPS_short-form-popularity) | Link público de CSV acessível. Ainda é necessário verificar horizonte, coleta, repetição de canais e acesso ao conteúdo. |
| [Shorts, mxmprhd](https://huggingface.co/datasets/mxmprhd/youtube-shorts-dataset) | 1.077 vídeos/48 criadores; somente ponteiros, contagens em um snapshot sem data por observação. Não sustenta retrospectivamente histórico disponível antes de publicar ou alvo em idade fixa. |
| [TikTok 5,6 bilhões, datasocial](https://huggingface.co/datasets/datasocial/tiktok-5.6B-videos) | Esquema público não inclui o autor do vídeo; contagens em `stats_updated_at`, sem séries por vídeo. Volume não resolve a ausência de histórico. |
| [Global YouTube Trending 2022–2025](https://doi.org/10.13012/B2IDB-9307654_V1) | Fonte acadêmica com quatro snapshots/dia. Repositório retornou HTTP 403 nesta sessão. Sem contornar restrições. |
| [TimePop, 2026](https://doi.org/10.1007/s44163-026-02322-9) | 2.200 vídeos/15 criadores; dados somente mediante solicitação aos autores. Não enviamos mensagens em nome do usuário. |
| [Open Channel Stats](https://openchannelstats.com/channels) | Cinco canais públicos nesta consulta; útil para auditoria externa pequena, insuficiente para confirmar desempenho nos setores do case. |

## Regras antes de ler desempenho da fonte Trending nova

- Países fixados: BR, US, GB, CA, DE, FR, MX. Categorias principais: Autos & Vehicles (2), Howto & Style (26). Deduplicar o mesmo vídeo entre países; não contar aparições como vídeos independentes.
- Alvo: primeiro snapshot com idade entre 168 e 192 horas. Sem interpolar views, inventar zeros ou aceitar contagem de idade arbitrária. Disponibilidade: instante observado acrescido de 24 horas conservadoras.
- Conteúdo textual: primeiro título registrado, com ressalva de ser posterior à publicação. Não usar likes, views iniciais, comentários, posição nas tendências nem tags atualizadas como atributos.
- Treino anterior a 2023; desenvolvimento em 2023, com ajustes que respeitem a maturação dos resultados. Teste confirmatório proposto: publicações de janeiro a março de 2024. Resultados finais ficam reservados. Metadados de disponibilidade/tamanho podem ser auditados sem comparar previsões.
- Referência histórica: selecionar entre mediana e média em log das últimas 5, 20 e 50 publicações maduras. Contexto recebe os mesmos atributos históricos que conteúdo.
- Hipótese de modelo: aprender diferenças entre pares do mesmo canal; comparar com regressão residual e regras simples. Seleção em desenvolvimento, não no teste. Não exigir que uma hipótese vença para publicá-la.
- Confirmação exige pelo menos 500 vídeos e 50 canais, ganho de erro de pelo menos 5% contra histórico e contexto e IC95% pareado por canal positivo; ordenação exige ganho sobre ambos, pelo menos 200 pares e 30 canais. Avaliar ordenação separadamente do erro numérico.
- Uma melhoria apenas no ranking não será descrita como melhoria do número de views. Uma vitória em Trending não será descrita como validação de campanhas, causalidade ou de todos os Shorts.

Este documento registra intenção e critérios antes da aquisição/modelagem. A decisão de executar a confirmação depende da auditoria. Novas descobertas e revisões serão acrescentadas, preservando o registro anterior.

## Revisão após auditar disponibilidade, antes de treinar

O download de sete países trouxe 1.881.232 snapshots. Há 13.189 vídeos nas categorias desejadas, mas somente **1.998** atendem às regras de idade, consistência e período até março de 2024. Desses, **922 vídeos de 119 canais** têm ao menos três resultados anteriores maduros. A auditoria consultou datas, identidade e consistência das contagens; não comparou previsões nem resumiu desempenho final.

O recorte originalmente proposto de janeiro–março de 2024 é pequeno demais. Em vez de reduzir o critério de sucesso, **todos os vídeos modernos passam a ser teste externo**. Ajuste e escolha ficam exclusivamente nas duas janelas de desenvolvimento de 2016 do estudo anterior, sem suas publicações do teste fechado. Ajuste final usa resultados disponíveis antes de 08/08/2016; calibração usa publicações de 08–10/08/2016. Nenhum parâmetro será ajustado usando rótulos de 2020–2024. A mudança é declarada antes de medir desempenho.

Essa é uma prova mais difícil de transferência entre épocas e seleções de dados. O alvo de origem cobre sete datas de calendário; o externo é uma contagem entre 168 e 192 horas segundo o timestamp nominal da fonte. Não são horizontes idênticos. O horário real da coleta pode divergir do timestamp nominal. Os históricos podem incorporar resultados externos já maduros, mas o modelo e as regras permanecem congelados.

**Modelos comparados:** 14 regressores residuais e 10 ordenadores diretos, além de oito históricos. As famílias usam somente contexto, texto lexical, semântica e comparáveis. A nova família de analogias consulta quanto vídeos semanticamente semelhantes ficaram acima/abaixo do próprio histórico, em vez de herdar o tamanho absoluto de outro canal. Comparáveis de outros canais vêm apenas do treino, já maduros antes de cada consulta; os locais respeitam a mesma disponibilidade. Ordenadores aprendem diferenças entre pares do mesmo canal/semana, com pesos equilibrados por canal. A ordenação e a previsão numérica têm seleções separadas e referências fortes próprias.

Duração, idioma informado pela plataforma e definição do vídeo não existem na fonte externa; foram removidos das entradas **em ambas as bases**. Não usamos uma duração inventada. Não há análise de imagem ou fala neste teste. A baixa quantidade de pares por semana será reportada; não alargaremos silenciosamente a janela para afirmar confirmação de ranking.

**AMPS auditada:** 13.011 IDs únicos, 48 canais, duas datas de coleta (31/07 e 22/08/2023), sem observações repetidas por vídeo; 12.611 vídeos publicados antes da primeira coleta. Isso não fornece os alvos e históricos em idade fixa necessários. Arquivo de origem acessível, não escolhido para fabricar um ganho temporal.

**Video Ads reavaliada:** 5.052 anúncios e 3.871 séries; cada série contém apenas uma lista `views`, sem datas ou índices diários explícitos. Recuperar o canal pelo ID público não resolveria a ausência de ancoragem temporal verificável e de investimento em mídia. Sem nova confirmação baseada em suposições sobre esses campos.

## Segunda rodada de desenvolvimento, ainda sem teste externo

A primeira rodada reduziu o erro de desenvolvimento de 0,5342 para 0,5280 (1,2%); o melhor ordenador continuou sendo uma regressão semântica (54,6%), não o modelo especializado em pares. A nova família de analogias tinha pouco suporte no primeiro treino: com maturação de nove dias e apenas duas semanas anteriores, quase nenhum exemplo de treino possuía comparáveis já maduros que também tivessem histórico próprio. Complexidade não criou informação ausente.

Revisão: aproveitar os dados anteriores a 20/08/2016, mantendo **inteiramente excluídas as publicações do teste encerrado de 20–31/08**. Desenvolvimento em 01–07/08 e 08–14/08; cada treino usa somente resultados maduros antes de começar sua validação. Ajuste final com resultados disponíveis antes de 17/08; calibração com publicações de 17–19/08, que amadurecem muito antes da avaliação externa de 2020–2024. Mantêm-se candidatos, métricas e critérios de sucesso. A [primeira seleção](results/transfer-development-v1/selection.json) e seu [plano](results/transfer-development-v1/plan.json) ficam preservados. Isso é mais desenvolvimento, não uma segunda confirmação.

Fontes adicionais: [WebShorts/Shorts-Cast](https://github.com/ryang1119/Shorts-cast), artigo de maio de 2026 com alvo de sete dias, ainda contém somente um README anunciando publicação futura (commit `d08729a`). [AdsTrace](https://huggingface.co/datasets/Xiuze/AdsTrace) é acessível e oferece 2.833 anúncios de beleza com áudio, frames a 1 FPS, transcrições, produto e conversão. Falta publicação/canal; é candidata a uma avaliação **complementar de conteúdo e conversão**, não substituta da comprovação de alcance sobre histórico.
