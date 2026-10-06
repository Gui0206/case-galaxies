# Decisões técnicas e correspondência com o planejamento

| Decisão | Por quê | Limite |
|---|---|---|
| Competição Ridge/CatBoost/LightGBM/blend | Seleciona por ranking, erro e complexidade em várias janelas | Superioridade em simulação não escolhe a família de produção |
| Alvos absolutos e residuais | Testa a hipótese de aprender desempenho relativo ao canal | A referência não remove confundimento causal |
| Quantis CatBoost + calibração por setor | Intervalos avaliados por cobertura, largura e interval score | Coortes pequenas usam correção global; canais novos ainda falham |
| Texto por hashing + SVD | Reprodutível, local e sem download obrigatório | Representação lexical/latente, não um encoder de linguagem de fronteira |
| CLIP opcional | Permite enriquecer o vídeo sem acoplar pesos pesados ao núcleo | Pesos e avaliação não fazem parte do teste básico |
| LangGraph determinístico | Fluxo inspecionável com contratos fixos | Não é uma conversa aberta ou persona sintética |
| Explicação por template + SHAP | Mantém os números e contribuições verificáveis | Não fornece interpretação subjetiva de segmentos de público |
| JSON/NPZ/CBM/texto LightGBM | Inspeção e persistência sem pickle | Manifesto não substitui confiança na origem |
| Interface HTML/CSS/JS | Executa no mesmo processo, sem build de frontend ou CDN | Escala de uma aplicação de pesquisa local |
| Horizontes alinhados | Evita confundir idade do vídeo com desempenho | Exige observações apropriadas; snapshots antigos isolados não bastam |
| Seleção congelada + teste final único por dataset | Hashes de dados/código e registro exclusivo tornam a escolha auditável | Não impede adulteração deliberada fora do protocolo |
| Promoção explícita | Um novo treino não substitui silenciosamente a versão atual | Critérios mínimos não equivalem a aprovação comercial |

## Implementado e exercitado localmente

Contratos, geração sintética, auditoria, histórico ponto no tempo, treino das quatro famílias, alvos residuais, atribuições aditivas, calibração, competição temporal, controles negativos, comparação, API, MCP, fila, processamento básico de um vídeo gerado e interface visual. Testes executam um cliente MCP de verdade e um arquivo de vídeo de verdade, ambos locais.

## Implementado, dependente de provisionamento/dados

Coleta oficial com chave, extração Whisper/CLIP/OCR, rubrica por VLM local, rastreamento MLflow e aplicação do protocolo prospectivo com criativos reais. Os adaptadores existem, mas seu desempenho no domínio não foi validado só por implementar o código.

## Pesquisa adicional orientada por resultados

Embeddings temporais de vídeo completo, ensembles com pesos aprendidos fora da amostra, modelos hierárquicos aprendidos e calibração dedicada a novos canais, interpretação generativa de persona e testes causais de campanha permanecem candidatos de pesquisa. O pipeline oferece pontos de extensão; não há promessa de que esses candidatos superem a configuração implementada.

Essa escolha preserva o princípio do planejamento: complexidade adicional precisa demonstrar ganho. “Sem prazos” não transforma experimentos sem dados em evidência de qualidade.

## Fora da entrega local

Construção de personas, conexão ao ambiente privado da Galaxies, acesso a dados de clientes sem fornecimento/autorização, publicação de campanhas, decisões de orçamento e produto SaaS multi-tenant.

A execução atual e suas falhas estão no [case de modelagem](MODEL_CASE.md). O teste de novos canais não foi omitido: os intervalos falharam nessa coorte sintética e as recomendações continuam bloqueadas.
