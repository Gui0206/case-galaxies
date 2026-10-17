# Revisão: prever alcance antes da publicação

Este estudo volta ao objetivo do TAPI e do kickoff: estimar visualizações e ordenar criativos do mesmo canal **antes da publicação**. Os três testes anteriores continuam encerrados; nenhum deles será reutilizado como confirmação.

## O que muda

1. **Alvo correto:** visualizações na data de publicação e nas seis datas seguintes, exigindo sete observações diárias explícitas. Isso é uma semana de calendário, não exatamente 168 horas.
2. **População mais próxima:** categorias públicas Autos & Vehicles e Howto & Style. A segunda contém outros tutoriais além de moda/beleza; não será renomeada como se fosse exclusivamente esses setores.
3. **Histórico forte:** comparar várias referências simples no desenvolvimento; só usar resultados de vídeos anteriores cujo horizonte já terminou. A disponibilidade é convencionada como data de publicação + 9 dias, incluindo margem para fuso e atraso do relatório.
4. **Aprender diferenças dentro do canal:** prever o resíduo em log sobre o histórico; acrescentar duração relativa, texto e similaridade com publicações anteriores. A hipótese é que comparáveis do próprio canal sejam mais úteis que um embedding isolado.
5. **Escolher por erro com amostra suficiente:** MAE em log, com cada canal tendo o mesmo peso. Ordenação é uma confirmação separada; poucos pares não decidirão a configuração.

## Fonte escolhida e alternativas auditadas

[Beyond Views, ICWSM 2018](https://github.com/avalanchesiqi/youtube-engagement), [Harvard Dataverse, DOI 10.7910/DVN/L3UWZT](https://doi.org/10.7910/DVN/L3UWZT): metadados, IDs de canal, datas de publicação e séries diárias. O arquivo público `tweeted_videos1.tar.bz2`, ID 3489832, contém as duas categorias. O registro declara CC0 1.0. O hash MD5 publicado é `1f01676fb6abd3ddc47e1289d64a2803`.

**Limitações conhecidas antes de treinar:** vídeos publicados em julho/agosto de 2016; amostrados por menções no Twitter, disponibilidade pública de estatísticas e pelo menos 100 visualizações em 30 dias. Não representam todos os uploads nem Shorts atuais. O arquivo não fornece os vídeos originais. Títulos/durações são retrospectivos; não há garantia de identidade com a versão pré-publicação. Distribuição paga é desconhecida. O histórico do experimento é o histórico disponível na amostra. Uma vitória aqui será evidência retrospectiva neste recorte, não validação de campanhas da Galaxies em 2026.

| Fonte examinada | Decisão antes de medir desempenho |
|---|---|
| [YouTube Videos With View Count Time Series, Zenodo](https://zenodo.org/records/14602718) | Não usar: o código público preserva datas de observação, mas não a publicação, e guarda nome de canal em vez de ID. Não inventar a cronologia. |
| [SMP-Video](https://huggingface.co/datasets/smpchallenge/SMP-Video) | Acesso depende de autenticação e aceite do titular; não foi acessada. |
| [Open Channel Stats](https://openchannelstats.com/channels) | Exportações autorizadas e atuais, mas apenas cinco canais públicos na consulta; população e tamanho não sustentam o teste principal dos setores. |
| [HIP / ACTIVE](https://github.com/andrei-rizoiu/hip-popularity) | Série diária disponível, mas seleção por atividade e população menos próxima; não escolhida. |
| Bases dos estudos anteriores | Testes fechados. Não selecionar um novo vencedor olhando os mesmos resultados. |

## Reserva de dados, antes de ler os resultados

As datas abaixo são fixadas a partir do período declarado pela fonte, antes da auditoria dos valores de visualizações:

- Desenvolvimento 1: publicações de 25 a 31 de julho de 2016; treino com resultados disponíveis antes de 25/07.
- Desenvolvimento 2: publicações de 1 a 7 de agosto; treino com resultados disponíveis antes de 01/08.
- Calibração: publicações de 8 a 10 de agosto; modelo ajustado apenas com resultados disponíveis antes de 08/08. O primeiro instante do teste fica em 20/08, depois da maturação de toda a calibração.
- Teste final: publicações de **20 a 31 de agosto de 2016**, aberto uma única vez depois de congelar seleção, dados e código.
- Canais reservados: 10% definidos por SHA-256 do ID, separados de todo ajuste e seleção. Serão relatados separadamente, sem misturar o resultado com canais conhecidos.

O teste principal exige ao menos três publicações anteriores com alvos maduros do canal; os casos com pouco histórico são relatados à parte. Duplicatas de ID e de título normalizado dentro do canal são removidas antes das partições, mantendo a primeira. Não se excluem vídeos por serem difíceis de prever. Alvos ausentes não viram zero; textos, categorias e datas não são inventados.

## O que contará como ganho

O candidato com conteúdo deve reduzir pelo menos **5% do MAE macro em log** em relação tanto à melhor referência histórica quanto ao modelo escolhido só com contexto; o limite inferior do intervalo pareado de 95%, reamostrado por canal, deve ser positivo para ambas as comparações. São necessários pelo menos 500 vídeos e 50 canais no teste principal.

Para afirmar que ajuda a **escolher criativos**, também deverá haver ganho de ordenação dentro do canal sobre o contexto, com limite inferior positivo e pelo menos 200 pares em 30 canais. Empate de previsão vale meio acerto; empate real não permite ordenar. Pares compartilham vídeos e a incerteza é calculada por canal, não como se fossem independentes.

Todas as configurações, inclusive as que falharem, serão registradas. Modelos simples podem vencer. Ganho só sobre uma referência fraca não basta. Os intervalos de previsão serão calibrados em dados separados e avaliados por cobertura e largura; não são garantia sob mudança de distribuição.

**Estado ao registrar este plano:** aquisição em andamento; nenhum modelo treinado ou resultado de teste consultado nesta nova base. A camada de vídeo completo existente permanece executável, mas seu ganho sobre visualizações ainda não foi demonstrado.
