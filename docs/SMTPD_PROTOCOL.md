# SMTPD: alcance em sete dias com histórico e conteúdo

Registro anterior à comparação preditiva desta fonte. Os testes anteriores continuam publicados, inclusive os negativos.

## Pergunta

Antes de publicar um vídeo de beleza/moda ou automotivo, analisar seu conteúdo melhora a previsão de visualizações na primeira semana em relação ao histórico do próprio canal e a um modelo que já conhece contexto e duração?

## Dados que realmente existem

A [SMTPD, CVPR 2025](https://arxiv.org/abs/2503.04446), acompanha vídeos a cada 24 horas desde a publicação. Usamos o CSV público dos [autores](https://github.com/zhuwei321/SMTPD), nas categorias Howto & Style e Autos & Vehicles. O escore do dia sete é invertido conforme a fórmula do artigo; todo valor deve corresponder a uma contagem inteira válida.

As datas e os IDs estáveis dos canais foram recuperados pela YouTube Data API v3. Não usamos contagens atuais, inscritos atuais nem campos numéricos padronizados globalmente pela fonte. Dos 12.655 IDs sem duplicatas, **8.112 vídeos de 5.291 canais** têm metadados públicos utilizáveis. A API foi consultada com uma chave restrita, guardada fora do Git.

O histórico só recebe um resultado depois da publicação acrescida de **oito dias**: sete de horizonte nominal e um dia de margem. Os horários exatos de observação não estão no CSV; essa convenção conservadora não substitui os logs originais.

## Separação antes de medir desempenho

| Uso | Publicação / disponibilidade | Tamanho relevante |
|---|---|---:|
| Validação 1 | Publicações de 17 a 27/11/2023; treino com resultados maduros antes de 17/11 | 77 vídeos, 34 canais |
| Validação 2 | Publicações de 18 a 22/12/2023; treino com resultados maduros antes de 18/12 | 169 vídeos, 73 canais |
| Treino final | Publicações anteriores a 23/12, com resultados maduros antes de 05/01 | 796 vídeos com pelo menos um resultado anterior |
| Calibração | Publicações de 23 a 27/12; todos os alvos maduros antes de 05/01 | 129 vídeos com histórico |
| **Teste reservado** | **Publicações a partir de 05/01/2024** | **515 vídeos, 165 canais** |

A avaliação exige pelo menos três resultados anteriores maduros. Histórias podem atualizar durante o teste somente quando cada resultado anterior já estiver maduro. Treino, vocabulário, escalas, redução dimensional e escolha de modelo permanecem fixos. A calibração é retrospectiva, com IDs separados do treino; não alegamos que suas previsões foram publicadas naquela época.

## Estratégia de modelagem

1. Comparar históricos fortes: mediana, média em log, janelas de 5/20/50 e suavização.
2. Acrescentar duração, horário, distância para publicações anteriores, tendência e comparáveis por duração. Esse é o controle de contexto, para não atribuir ao conteúdo um ganho que a duração já explica.
3. Testar texto lexical e semântico, similaridade com vídeos anteriores e regressão de diferenças dentro do mesmo canal. Priorizar modelos regularizados e correções moderadas ao histórico.
4. Extrair imagem, fala e edição dos **669 vídeos de até 180 segundos** que atendem ao plano de treino/avaliação. São 7,38 horas de mídia; **222 vídeos curtos** estão no teste. O restante continua na avaliação principal com a previsão textual/contextual, sem inventar atributos de mídia.
5. Escolher configurações usando apenas as duas validações. Publicar hashes e seleção antes de abrir o teste uma única vez.

A coleta é sequencial, preserva falhas e interrompe em bloqueios de acesso. Os arquivos não são cortados. Whisper analisa o áudio completo; OpenCLIP recebe imagens distribuídas pelo vídeo; o detector de edição decodifica todos os frames. Sem alegar que todas as imagens passam pelo encoder semântico.

## O que conta como evidência

Erro primário: MAE de `log(1 + views)`, com peso igual para cada canal. Confirmação exige pelo menos 500 vídeos/50 canais, redução de pelo menos **5%** e IC95% pareado por canal inteiramente positivo **contra histórico e contexto**. Ordenação é avaliada separadamente e exige pelo menos 200 pares/30 canais. Intervalos de previsão são calibrados em dados separados; reportar cobertura e largura.

O teste de vídeos curtos é uma ablação exploratória menor, mesmo se apresentar números melhores. A avaliação principal preserva todos os 515 vídeos, inclusive aqueles sem mídia. Não reduziremos critérios ou escolheremos um subgrupo depois de observar o resultado para declarar vitória.

## Limites que um bom resultado não apagaria

A fonte tem lacunas no histórico, exclui vídeos com dias ausentes e extremos de popularidade. Somente vídeos ainda públicos puderam ser recuperados. Howto & Style inclui assuntos além de moda e beleza. A mídia recuperada agora pode ter sido editada desde a coleta. Likes no mesmo horizonte e validação prospectiva na população da Galaxies continuam sem base adequada. Uma vitória aqui não equivale a causalidade, garantia de viralização ou validação de campanhas.
