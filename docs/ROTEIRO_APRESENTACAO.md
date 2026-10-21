# Roteiro de apresentação — Galaxies

Este roteiro acompanha a apresentação **Galaxies — conteúdo, histórico e evidência**. Os slides 1 a 10 formam a apresentação principal, com duração aproximada de 8 a 10 minutos. Os slides 11 e 12 são materiais de apoio para perguntas da banca.

## Mensagem central

O projeto entregou uma ferramenta funcional e avaliou, com dados reais e teste reservado, se analisar o conteúdo do vídeo melhora a previsão de visualizações além do histórico e do contexto do canal. O ganho não foi confirmado. A decisão profissional foi preservar a ferramenta, registrar o resultado negativo e não promover o modelo como superior.

A formulação mais segura durante toda a apresentação é:

> Nesta base, com estas características e este protocolo, o conteúdo não demonstrou ganho confiável sobre histórico e contexto.

Evite dizer que “o conteúdo não ajuda”, que “o modelo não funciona” ou que “mais dados certamente resolveriam”. Essas afirmações vão além da evidência.

---

## Slide 1 — Conteúdo, histórico e evidência

**Tempo sugerido:** 30 segundos

### Fala

> Este projeto investiga uma pergunta simples e importante para a Galaxies: antes de publicar um vídeo, conseguimos prever seu desempenho e escolher melhor os conteúdos?
>
> Eu construí uma ferramenta funcional e, principalmente, testei se analisar o conteúdo acrescenta informação além do histórico do canal. Vou mostrar os dados utilizados, como fizemos uma avaliação temporal e o que os resultados realmente permitem concluir.

### Transição

> Primeiro, precisamos definir exatamente qual decisão queremos apoiar.

---

## Slide 2 — A decisão antes da publicação

**Tempo sugerido:** 45 segundos

### Fala

> O alvo foi o número nominal de visualizações na primeira semana. A previsão acontece antes da publicação, portanto as primeiras visualizações do próprio vídeo não podem entrar como atributo.
>
> O desafio também não é apenas produzir um número. O modelo precisa superar referências úteis: o desempenho histórico do canal e uma referência mais forte, que combina esse histórico com contexto disponível antes da publicação.
>
> Essa comparação evita confundir o tamanho do canal com a qualidade ou a adequação do criativo.

### Transição

> Com a pergunta definida, o próximo passo foi encontrar dados reais compatíveis com esse horizonte.

---

## Slide 3 — Dados reais, com limites conhecidos

**Tempo sugerido:** 60 segundos

### Fala

> A principal fonte foi a SMTPD, uma base pública apresentada na CVPR 2025, com trajetórias de popularidade de vídeos do YouTube. Recuperamos datas e identidades dos canais pela API oficial.
>
> Dos 12.655 IDs únicos investigados, 8.112 tinham metadados utilizáveis. O teste final ficou com 515 vídeos de 165 canais.
>
> A base é real, mas tem limites: a amostra foi selecionada, os históricos dos canais são incompletos e a categoria Howto & Style é mais ampla que moda e beleza. Também não representa campanhas atuais da Galaxies.
>
> Por isso, trato o resultado como evidência real para o case, mas não como validação de produção no domínio da empresa.

### Se perguntarem onde estão os 5.327 vídeos

> O estudo tinha 5.327 vídeos disponíveis para algum tipo de treinamento, incluindo exemplos sem histórico suficiente para o modelo principal. O treino principal comparável usou 796 vídeos com histórico maduro. Não somamos esses números porque eles cumprem papéis diferentes no protocolo.

### Transição

> Além de usar dados reais, precisávamos impedir que o modelo enxergasse informações futuras.

---

## Slide 4 — Avaliação temporal com teste reservado

**Tempo sugerido:** 60 segundos

### Fala

> Organizamos a avaliação no tempo. O treino principal teve 796 vídeos com histórico disponível. Outros 129 exemplos serviram para calibrar os intervalos de previsão, e 515 vídeos ficaram reservados para o teste final.
>
> Cada publicação anterior só podia entrar no histórico depois que seu resultado de sete dias já estivesse maduro. Usamos oito dias como margem conservadora.
>
> A escolha do modelo foi congelada antes da abertura do teste. Depois, uma implementação independente recalculou as métricas a partir das previsões publicadas.
>
> Esse desenho não garante que o modelo será bom, mas torna a conclusão mais confiável e reduz o risco de ajustar a solução ao resultado que queríamos encontrar.

### Transição

> Dentro desse protocolo, testamos contexto, tema e características do vídeo completo.

---

## Slide 5 — O que testamos com dados reais

**Tempo sugerido:** 75 segundos

### Fala

> A previsão começa no histórico do canal e acrescenta contexto, como duração e momento da publicação. Para representar o tema, comparamos o título do novo vídeo com publicações anteriores do mesmo canal, por palavras e por embeddings semânticos.
>
> No vídeo completo, extraímos três grupos de sinais. Na imagem, usamos OpenCLIP em frames distribuídos pelo vídeo e nos primeiros segundos. Na fala, usamos Whisper para transcrever o áudio integral e representar semanticamente o que foi dito. Na edição, medimos cortes, duração dos planos, movimento, brilho e saturação.
>
> Também medimos sinais da abertura, como cortes, movimento e densidade de fala nos primeiros três segundos. Isso descreve o gancho, mas não constitui uma nota humana validada de qualidade do gancho.
>
> Tema combinado com histórico já foi explorado. A rubrica VLM, que produziria julgamentos mais interpretativos sobre a estrutura criativa, permaneceu sem validação real.

### Explicação simples de “variação visual”

> Comparamos representações de frames ao longo do vídeo. Um vídeo com a mesma pessoa no mesmo cenário tende a variar menos; um vídeo que alterna rosto, produto e ambiente tende a variar mais. Isso mede mudança visual, não qualidade.

### Explicação simples de “semelhança com conceitos”

> O OpenCLIP compara as imagens com descrições, como “pessoa falando para a câmera”, “demonstração de produto” ou “carro na estrada”. O resultado é uma semelhança numérica, não uma confirmação nem uma probabilidade. No caso de texto na tela, ele não lê o conteúdo escrito.

### Transição

> Depois de extrair tudo isso, a pergunta decisiva foi se o erro realmente diminuiu.

---

## Slide 6 — O ganho sobre o histórico não foi confirmado

**Tempo sugerido:** 75 segundos

### Fala

> Este é o resultado principal. Menor erro é melhor. O modelo teve erro 0,18% menor que o histórico puro, mas ficou 0,03% pior que a referência de histórico mais contexto.
>
> As diferenças são muito pequenas e os intervalos de incerteza cruzam zero. Portanto, não demonstramos uma vantagem confiável.
>
> A métrica principal é o erro absoluto na escala logarítmica, calculado primeiro por canal. Ela evita que poucos vídeos gigantes dominem a avaliação. Esses percentuais representam redução de erro, não aumento de visualizações.
>
> A decisão profissional é não promover o modelo para produção nem afirmar que ele supera o histórico.

### Como explicar o resultado sem parecer que o case fracassou

> O objetivo de comprovar superioridade preditiva não foi atingido. A investigação, porém, respondeu à pergunta com dados reais, evitou uma conclusão artificial e entregou uma ferramenta que mostra exatamente o que foi e o que não foi validado. Para a empresa, descobrir que uma solução mais complexa ainda não justifica adoção também evita uma decisão ruim.

### Frase curta para reforçar

> O modelo produz previsões; o que ele não demonstrou foi superioridade sobre uma referência forte.

### Transição

> Também isolamos o efeito das modalidades do vídeo para entender se o custo multimodal se justificava.

---

## Slide 7 — Imagem e fala não demonstraram ganho adicional

**Tempo sugerido:** 60 segundos

### Fala

> A análise multimodal usou 219 vídeos completos de 63 canais, todos com até três minutos. Comparamos texto, imagem, fala, edição e a combinação completa usando as mesmas referências.
>
> A combinação de todas as modalidades ficou cerca de 0,5% pior que a referência textual. Nenhuma modalidade apresentou uma vantagem confirmada.
>
> Esse recorte é exploratório e menor que o teste principal. Portanto, a conclusão vale para os extratores e modelos avaliados. Ela não permite afirmar que imagem ou fala nunca ajudam.
>
> O aprendizado é mais específico: adicionar complexidade e custo de processamento não trouxe benefício preditivo demonstrado nesta implementação.

### Transição

> Essa limitação aparece diretamente na experiência da ferramenta.

---

## Slide 8 — A interface separa os usos sustentados pelos dados

**Tempo sugerido:** 60 segundos, sem demonstração; 90 segundos, com demonstração

### Fala

> A interface tem dois fluxos separados.
>
> Em “Meu vídeo”, o usuário envia um arquivo e recebe uma análise descritiva: transcrição, medidas de edição e características visuais. A ferramenta não inventa uma previsão de views para um canal sem histórico e não atribui uma nota de qualidade não validada.
>
> Em “Canais da base”, reproduzimos exemplos históricos com previsão, histórico, intervalo de incerteza e resultado observado. Esse fluxo demonstra os modelos reais congelados, sem treinar novamente.
>
> A interface transforma o limite científico em uma decisão de produto: mostrar somente o que os dados sustentam.

### Demonstração sugerida

1. Abra “Meu vídeo” e mostre os resultados descritivos.
2. Diga explicitamente que não há previsão de views para um canal desconhecido.
3. Abra “Canais da base” e selecione um exemplo histórico.
4. Mostre histórico, previsão, intervalo e resultado observado.

### Se a demonstração não abrir

> A ferramenta depende dos pesos e caches locais, que não acompanham apenas o clone do Git. Para a apresentação, o exemplo do próximo slide preserva a demonstração sem depender de conexão ou treinamento.

### Transição

> Este exemplo mostra por que a incerteza precisa aparecer para o usuário.

---

## Slide 9 — Um exemplo real expõe a incerteza

**Tempo sugerido:** 60 segundos

### Fala

> Este é um exemplo real escolhido pela ordem dos dados, sem selecionar um caso favorável ao modelo.
>
> O histórico previa aproximadamente 80 mil visualizações, o modelo cerca de 56 mil e o resultado observado foi 115 mil. Neste caso, o histórico ficou mais próximo.
>
> O intervalo nominal de 90% vai de aproximadamente 12 mil a 268 mil views. Ele contém o resultado, mas é amplo demais para sustentar uma decisão precisa de investimento.
>
> O resultado observado só é anexado depois da inferência. Ele nunca entra como atributo da previsão.

### Transição

> Com o teste aberto e sem ganho confirmado, precisávamos decidir se valia continuar tentando.

---

## Slide 10 — Por que encerramos esta etapa

**Tempo sugerido:** 75 segundos

### Fala

> Antes de propor outra rodada, revisamos o que já havia sido testado. Tema combinado com histórico já tinha implementações lexicais e semânticas. Houve promessa no desenvolvimento, mas ela não se confirmou no teste final.
>
> Também revisamos a ordenação dentro do canal e o recorte com pouco histórico. Nenhum deles revelou uma oportunidade clara para repetir a mesma ideia.
>
> Por isso, encerramos esta etapa. Preservamos a ferramenta, as previsões e os resultados negativos, mantendo histórico e contexto como referências.
>
> Esta decisão não significa que conteúdo nunca ajuda. Significa que repetir tentativas semelhantes, depois de observar o teste, teria baixo retorno e contaminaria a avaliação.
>
> Uma retomada precisa de uma hipótese materialmente diferente, desenvolvimento separado e novos dados reservados adequados ao domínio.

### Encerramento forte

> Minha entrega é uma ferramenta funcional e uma resposta auditável: nesta base, com estas características e este protocolo, o conteúdo não demonstrou ganho confiável sobre histórico e contexto. Em vez de vender uma acurácia que os dados não sustentam, a solução deixa claro o que funciona, o que permanece incerto e qual evidência seria necessária para avançar.

---

## Slide 11 — Apoio: ordenação dentro do canal

**Use somente se perguntarem se o objetivo deveria ser escolher o melhor criativo.**

### Fala

> Também avaliamos a decisão relativa: entre vídeos do mesmo canal publicados na mesma semana, qual teria melhor desempenho?
>
> Foram 726 pares de 54 canais. O histórico acertou 47,4%, contexto 56,8% e conteúdo 55,8%. O conteúdo não superou contexto, e o intervalo de incerteza não confirmou vantagem.
>
> Esses pares são observacionais. Eles não representam um experimento causal em que alteramos somente o roteiro e mantemos todas as outras condições iguais.

---

## Slide 12 — Apoio: evidências para encerrar

**Use se perguntarem por que não testamos mais uma característica.**

### Fala

> A hipótese tema combinado com histórico já foi testada. No desenvolvimento, a melhor recuperação semântica teve erro 0,663, enquanto contexto teve 0,646; menor é melhor.
>
> Também examinamos, de forma exploratória, os 200 vídeos com apenas três a cinco antecedentes maduros. Conteúdo teve erro 0,709 e contexto 0,707. Não apareceu vantagem nesse grupo.
>
> O intervalo existente contra contexto equivale aproximadamente a 2% de piora até 1,8% de melhora para o modelo avaliado. Um benefício pequeno continua possível, mas a meta de 5% ficou fora desse intervalo.
>
> Esses recortes não serviram para escolher um novo modelo. Eles apenas sustentam a decisão de não repetir uma tentativa semelhante nesta entrega.

---

# Perguntas prováveis da banca

## “Então o case fracassou?”

> O objetivo de comprovar ganho preditivo não foi atingido. A entrega técnica e a investigação foram concluídas: construímos a ferramenta, avaliamos a hipótese com dados reais e documentamos uma decisão de não adoção. Eu não apresentaria o modelo como pronto para orientar investimento, mas a empresa ganha uma conclusão confiável e evita adotar complexidade sem benefício demonstrado.

## “O modelo funciona ou não?”

> Ele funciona tecnicamente e produz previsões reproduzíveis. O que não demonstrou foi desempenho superior ao histórico e ao contexto. Por isso, não foi promovido para produção.

## “Por que o modelo não foi melhor?”

> Não podemos atribuir o resultado a uma causa única. O histórico do canal já é uma referência forte, o sinal adicional do conteúdo pode ser pequeno, as características avaliadas podem não capturar os fatores mais relevantes e a base tem limitações. Os dados não permitem afirmar qual dessas explicações domina.

## “Mais dados resolveriam?”

> Mais dados adequados poderiam reduzir a incerteza e melhorar o desenvolvimento, mas não garantem ganho. O modelo ainda precisaria extrair informação útil além do histórico. Por isso, eu trataria novos dados como condição para uma avaliação melhor, não como promessa de sucesso.

## “Por que não acompanhar apenas as views do primeiro, segundo e terceiro dia?”

> Isso mudaria a pergunta. Views iniciais são muito úteis para prever views futuras, mas só existem depois da publicação. O case procura apoiar a decisão antes de publicar. Para o alvo de sete dias, precisamos do conteúdo e do contexto disponíveis antes da publicação e da contagem observada quando o vídeo completa sete dias; não precisamos obrigatoriamente dos dias intermediários.

## “Qual base de dados foi usada?”

> A principal foi a SMTPD, uma base pública de vídeos do YouTube apresentada na CVPR 2025. Recuperamos metadados pela API oficial. Tivemos 8.112 vídeos com metadados utilizáveis, teste final de 515 vídeos de 165 canais e 219 vídeos completos no recorte multimodal. Ela é real, mas não representa campanhas atuais da Galaxies.

## “Os dados eram sintéticos?”

> O repositório preserva um protocolo sintético antigo para verificar implementação, mas os resultados apresentados como evidência principal usam dados reais da SMTPD. Os resultados sintéticos ficam separados e não sustentam alegações sobre campanhas reais.

## “Como vocês garantiram que não houve vazamento de informação futura?”

> A divisão foi temporal. Cada publicação anterior só entrou no histórico depois que seu resultado estava maduro. Escolhemos e congelamos o modelo antes de abrir o teste, e uma verificação independente recalculou as métricas das previsões publicadas.

## “Por que usar erro em log?”

> Visualizações têm uma distribuição muito desigual. Alguns vídeos têm ordens de grandeza a mais que outros. A transformação logarítmica reduz o domínio desses extremos e mede melhor o erro multiplicativo. Também calculamos a média por canal para evitar que canais com muitos vídeos dominem o resultado.

## “0,18% menor que histórico significa 0,18% mais views?”

> Não. Significa uma redução de 0,18% no erro da previsão contra o histórico. O modelo não causa visualizações, e esse número não representa aumento de alcance.

## “Por que comparar também com histórico mais contexto?”

> Porque superar apenas uma média histórica simples pode significar que o modelo aprendeu duração ou horário, sem extrair valor do conteúdo. O contexto é uma referência mais forte e mais justa para medir o ganho incremental do conteúdo.

## “O intervalo cruzar zero significa o quê?”

> Significa que os dados são compatíveis tanto com uma pequena melhora quanto com uma pequena piora. Portanto, não temos evidência suficiente para afirmar vantagem.

## “O teste tinha poder suficiente?”

> O bootstrap por canal já mediu a incerteza do modelo avaliado. Contra contexto, o intervalo relativo aproximado vai de 2% de piora a 1,8% de melhora. Isso mostra que ganhos pequenos permanecem possíveis, mas a meta de 5% não foi sustentada. Não fizemos um cálculo formal de poder para dimensionar um novo estudo.

## “Por que escolher uma meta de 5%?”

> A meta representa um ganho mínimo relevante para justificar mais complexidade, custo e risco operacional. Uma diferença de terceira casa decimal, mesmo que real, dificilmente justificaria substituir uma referência simples sem outro benefício operacional.

## “Quais características foram extraídas do vídeo?”

> Na imagem, extraímos representações dos frames, abertura, variação visual e semelhança com conceitos. Na fala, usamos transcrição, significado, alinhamento entre título e fala, densidade, início da fala, volume e silêncio. Na edição, medimos cortes, duração dos planos, movimento, brilho e saturação. Também usamos título, duração, momento de publicação e histórico do canal fora do vídeo.

## “Vocês mediram a qualidade do gancho?”

> Medimos sinais objetivos dos primeiros segundos, como cortes, movimento, imagem e densidade de fala. Não validamos uma nota humana de clareza, promessa ou força do gancho. Portanto, não afirmamos ter medido sua qualidade criativa completa.

## “A imagem e a fala não ajudam?”

> Não demonstraram ganho com os extratores, modelos e amostra avaliados. Isso é diferente de provar que nunca ajudam. A conclusão está limitada à implementação testada.

## “Tema combinado com o histórico ainda valeria tentar?”

> Essa hipótese já foi explorada por palavras, embeddings semânticos e regressão dentro do canal. A promessa observada no desenvolvimento não se confirmou no teste. Eu só retomaria com uma formulação materialmente diferente e novos dados reservados.

## “Por que não continuar ajustando o modelo até melhorar?”

> Depois de abrir o teste, usá-lo repetidamente para escolher novas características transforma o teste em desenvolvimento e aumenta o risco de encontrar um ganho por acaso. A decisão correta foi preservar o resultado e exigir novos dados reservados para uma nova hipótese.

## “Por que havia apenas 796 vídeos no treino principal se a base era maior?”

> O modelo principal exigia histórico maduro do mesmo canal e respeitava os cortes temporais. Muitos vídeos estavam disponíveis para outras etapas, mas não atendiam a todos os critérios do treino comparável. Preferimos uma amostra menor e válida a usar informações incompatíveis com a decisão real.

## “O modelo funciona para um canal novo?”

> Não há validação de previsão de views para canal sem histórico. Para um vídeo próprio de canal desconhecido, a ferramenta entrega análise descritiva. Selecionar um canal da base como substituto criaria uma precisão aparente sem sustentação.

## “O que a ferramenta entrega hoje?”

> Para um vídeo próprio, entrega transcrição e medidas descritivas de imagem, fala e edição. Para exemplos históricos da base, reproduz previsão, histórico, intervalo e resultado observado. Ela não entrega probabilidade validada de viralização nem recomendação automática de investimento.

## “Qual foi o principal aprendizado de negócio?”

> O histórico do canal é difícil de superar e deve permanecer como referência. Processamento multimodal só merece custo adicional quando comprovar ganho incremental. A ferramenta também precisa comunicar incerteza e recusar previsões para cenários sem cobertura.

## “Qual seria o próximo passo correto?”

> Definir uma hipótese realmente diferente antes de coletar dados, registrar previamente a métrica e o ganho mínimo relevante, desenvolver sem tocar no próximo teste e avaliar em publicações novas do domínio. Sem canal parceiro, essa validação deve ser apresentada como plano futuro, não como resultado já disponível.

## “Existem artigos que conseguiram prever views?”

> Sim. Há trabalhos que relatam sucesso usando conteúdo, metadados sociais ou popularidade inicial. Os estudos com primeiras views respondem a uma tarefa diferente, pois observam o vídeo depois da publicação. Há também resultados antes da publicação, mas eles não garantem que a vantagem se reproduza nesta população, neste horizonte e contra estas referências.

## “O que você faria diferente se começasse novamente?”

> Eu anteciparia a auditoria dos dados e a comparação com referências simples. Também definiria critérios de interrupção antes de investir em infraestrutura extensa. Isso permitiria concentrar esforço mais cedo nas hipóteses com maior chance de ganho material.

## “Qual nota você dá ao case?”

> Eu separaria a avaliação em dimensões. A engenharia, o protocolo e a honestidade são fortes. A evidência preditiva ainda não sustenta adoção. O valor do case está em responder a pergunta sem esconder o resultado e em entregar uma base reproduzível para decisões futuras.

---

# Frases que ajudam durante a defesa

- “Ganho não demonstrado” é mais preciso que “o conteúdo não ajuda”.
- “O modelo produz previsões, mas não demonstrou superioridade sobre a referência.”
- “O resultado vale para esta base, estas características e este protocolo.”
- “A extração funcionar tecnicamente não comprova utilidade preditiva.”
- “A complexidade só se justifica quando acrescenta valor mensurável.”
- “O teste final serviu para decidir, não para iniciar outra busca de hiperparâmetros.”

# Frases a evitar

- “O modelo quase funcionou.”
- “Com mais dados certamente funcionaria.”
- “Imagem e fala não servem para prever views.”
- “O modelo aumenta as visualizações.”
- “O intervalo prova que a previsão está correta.”
- “A ferramenta funciona para qualquer canal.”
- “A base representa as campanhas da Galaxies.”

# Checklist para o dia da apresentação

- Abrir os slides antes de entrar na sala.
- Se for demonstrar a interface, confirmar que o servidor local e os pesos estão disponíveis.
- Não iniciar treino, coleta ou download durante a apresentação.
- Ter o slide 9 pronto como alternativa à demonstração ao vivo.
- Decorar os quatro números centrais: **515 vídeos**, **165 canais**, **0,18% menos erro que histórico**, **0,03% mais erro que contexto**.
- Encerrar com a decisão, sem pedir desculpas pelo resultado negativo.

