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
