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


## Congelamento da transferência externa

A segunda rodada terminou: erro de desenvolvimento **0,5449 no histórico**, **0,5403 no contexto** e **0,5313 no conteúdo com analogias** (redução de 2,5% sobre histórico). Para ordenar pares, o vencedor foi o modelo lexical treinado diretamente: **56,9%**, contra **50,5%** do histórico e **52,6%** do contexto. Esses resultados ainda são desenvolvimento.

O [plano final](results/transfer/plan.json) e a [seleção final](results/transfer/selection.json) ficam congelados antes da primeira avaliação externa de 2020–2024. Os 88 testes passaram. Não haverá escolha de outra configuração após abrir o teste.


## Resultado externo e decisão

O [teste de transferência](TRANSFER_RESULTS.md) terminou sem ganho: erro macro de conteúdo 0,4386, contra 0,4245 do histórico e 0,4194 do contexto. A ordenação também piorou, em uma amostra pequena de 40 pares/18 canais. As configurações não serão trocadas depois desse resultado. Esse teste passa a estar encerrado; pode informar hipóteses futuras, mas não poderá ser reapresentado como confirmação inédita.

**AdsTrace não será priorizada:** além de não conter canal/data de publicação, a unidade da coluna chamada CVR não está suficientemente esclarecida. Há valor acima de 1 no treino; o código dos autores apenas aplica log1p. São anúncios profissionais e faltam exposição/investimento. Baixar 10 GB de áudio/frames agora não responde ao objetivo principal de alcance sobre histórico. Preservamos a auditoria e voltamos a uma fonte mais adequada.

## SMTPD: alternativa que justifica nova investigação

A [SMTPD, CVPR 2025](https://arxiv.org/html/2503.04446v1) acompanha visualizações nos primeiros 30 dias. O [repositório dos autores](https://github.com/zhuwei321/SMTPD), commit `9658ea11aff81c533c46070b53b5bd90d1a29c70`, fornece CSV e miniaturas. O CSV público foi baixado integralmente (608.627.296 bytes); contém 282.481 linhas, 282.161 IDs distintos e 12.657 linhas nas duas categorias relevantes. A fonte filtra falhas, exclusões e outliers, o que limita representatividade.

Os campos `Day 1` a `Day 30` usam `log2(views/dia + 1)`, conforme a equação do artigo. A inversão do sétimo dia produz contagens compatíveis com inteiros nos 12.657 registros. Os campos numéricos já foram padronizados sobre a base; não serão utilizados. O arquivo omite publicação e usa apelidos/URLs de canal, com valores inválidos e duplicações: não permite montar um histórico seguro diretamente.

Uma sondagem determinística de 48 IDs, ordenados pelo SHA-256 sem consultar desempenho, recuperou metadados públicos em parte dos vídeos. A recuperação usa apenas identidade estável do canal, publicação, duração e condição de transmissão ao vivo. Não incorpora contagens atuais. A coleta ampliada encontrou HTTP 429 e foi interrompida. Foi preparada uma retomada sequencial após espera, com intervalo de dois segundos e parada imediata em nova restrição; sem cookies, proxies ou desvio de controles.

O alvo continua sendo alcance de sete dias, com margem conservadora de disponibilidade de oito dias. Vídeos privados, ausentes, lives, identidade/data inválida e IDs duplicados não serão preenchidos artificialmente. Nenhuma comparação preditiva da SMTPD foi feita nesta etapa. Caso a recuperação seja viável, os cortes de treino/validação/teste serão registrados a partir das datas e cobertura, antes de comparar modelos. Arquivos recuperados em 2026 não garantem que a edição do vídeo seja idêntica à original; essa limitação será explícita.
