# Dados que sustentam uma previsão

**Pergunta de controle:** essa informação estava disponível no momento em que a escolha seria feita?

## Bases públicas já testadas

O [estudo público de metadados](REAL_DATA_RESULTS.md) usa Ken Jee (196 vídeos elegíveis) e YouTube Trending (2.534 vídeos elegíveis). Fontes, versões, hashes, alvos por calendário e resultados estão documentados. **O modelo não demonstrou ganho consistente sobre o histórico.**

Essas bases não comprovam versões originais anteriores à publicação nem observações em exatamente 168 horas. Por isso são processadas por `public_benchmark.py`, separadamente do contrato abaixo. Não inventamos datas ou setores para importá-las como dados prontos para produção. Os CSVs ficam em `.runtime/public/`, ignorado pelo Git; não são a base sintética da interface.

O [estudo de vídeos completos](FULL_VIDEO_METHOD.md) usa uma terceira fonte, Tsinghua ShortVideo: arquivos de vídeo e logs com criador, duração e tempo assistido. O alvo é retenção no primeiro dia observado pelo painel, porque não há data de publicação verificável. `shortvideo.py` e `retention_benchmark.py` mantêm esse alvo separado do contrato de contagens; não preenchem a data de publicação com uma suposição. Vídeos completos, transcrições e logs permanecem fora do Git.

Foram aceitos 945 dos 954 arquivos solicitados: oito tinham duração incompatível e um estava incompleto na fonte. O [resultado reservado com 177 vídeos](FULL_VIDEO_RESULTS.md) não confirmou vantagem da análise completa sobre o histórico.

```mermaid
timeline
    title Disponibilidade da informação
    Antes de publicar : Arquivo e legenda originais : Estatísticas do canal conhecidas
    Publicação : Decisão e previsão congeladas
    Horizonte de 7 dias : Observação de curtidas
    Até 24 horas depois : Tolerância de observação do alvo
```

## Contrato de uma linha

Veja [exemplo completo](../examples/post-record.json) e [JSON Schema](schemas/post-record.json).

| Campo | Significado |
|---|---|
| `creative_id`, `channel_id` | Identificadores estáveis |
| `sector` | `beleza`, `automotivo` ou `moda` |
| `published_at` | Data de publicação com fuso |
| `features_available_at` | Quando o conteúdo/contexto original estava disponível |
| `observed_at` | Quando a métrica-alvo foi efetivamente medida |
| `metric`, `horizon_days`, `target` | Métrica, horizonte e contagem observada |
| `subscribers`, `channel_stats_observed_at` | Estatística do canal e quando ela era conhecida; ambos ausentes quando desconhecidos |
| `caption`, `transcript`, `features` | Conteúdo e atributos anteriores à publicação |
| `visual_embedding` | Vetor opcional produzido pelo mesmo encoder em toda a base |
| `evidence` | Observações com atributo e intervalo de segundos |
| `attributes_source` | `media_extraction`, `human_annotation` ou `simulation` |
| `content_hash`, `content_group` | Identidade do arquivo e grupo de campanha/variações |
| `source`, `usage_basis` | Origem e justificativa de uso dos dados |
| `synthetic` | Identifica dados artificiais explicitamente |
| `distribution` | `organic`, `paid` ou `unknown`; não inferir ausência de mídia paga |

`features_available_at` é a disponibilidade do **conteúdo original**, não o horário em que um algoritmo processou o arquivo. Analisar hoje um arquivo imutável comprovadamente disponível antes da publicação é diferente de usar hoje uma legenda que foi editada depois.

Declarar um tempo ou uma licença no JSON não prova sua veracidade. Essa proveniência precisa ser auditada na fonte.

Escalas dos atributos: `cta` e `demonstration` são binários (0 ou 1); as demais notas de rubrica variam de 0 a 10. Luminosidade, saturação, variação entre frames e proporção de silêncio usam 0–1. Duração em segundos precisa ser positiva; contagens, taxas e energia não podem ser negativas. Use `null` ou omita uma chave desconhecida.

## Regras automáticas

- Uma base por métrica/horizonte; não misturar curtidas e visualizações.
- Dados sintéticos e reais não se misturam.
- Atributos como `views`, `likes`, `comments` e `target` não são aceitos dentro de `features`.
- Inscritos exigem um timestamp anterior ou igual à publicação.
- O alvo deve ser medido entre `publicação + horizonte` e esse instante mais 24 horas.
- Ausência de uma métrica é desconhecimento, não zero.
- Duplicatas exatas são removidas antes da partição; grupos de conteúdo não atravessam partições.
- A anotação `content_group` precisa ser curada: não existe detecção automática de toda variação semântica/republicação.

## Formatos e importação

```bash
uv run galaxies validate minha-base.jsonl
uv run galaxies import-data minha-base.jsonl --name cliente-v1.jsonl
```

JSON contém uma lista. JSONL contém um objeto por linha. Parquet mantém objetos complexos (`features`, `visual_embedding`, `evidence`) serializados como JSON; `save_records` produz o formato correto. CSV de terceiros precisa ser adaptado para esse contrato, com semântica de datas e métricas preservada.

## Coletar observações oficiais

```bash
export YOUTUBE_API_KEY="sua-chave"
uv run galaxies snapshot VIDEO_ID_1,VIDEO_ID_2 snapshot-2026-10-06.json
```

Esse comando coleta **os números atuais**. Não baixa vídeo e não transforma contagens atuais em métricas históricas de sete dias. O coletor também oferece `channel_video_ids`, por playlist de uploads, para descoberta sem pesquisa repetida.

Crie um manifesto de criativos com os campos de `PostRecord`, exceto `metric`, `horizon_days`, `target` e `observed_at`. Consolide os snapshots numa lista JSON e execute:

```bash
uv run galaxies assemble manifesto.json observacoes.json base-7d.jsonl --metric likes --horizon-days 7
```

Registros sem observação no horizonte permanecem pendentes; o comando informa seus IDs. Perder a janela exige descartar esse alvo ou usar outro horizonte com observação válida. Não interpole retrospectivamente sem um experimento separado.

## Origem, retenção e acesso

Use arquivos originais de criadores que autorizaram seu processamento ou datasets com condições compatíveis. Consulte as [políticas da API](https://developers.google.com/youtube/terms/developer-policies) e as [condições adicionais para métricas/armazenamento](https://developers.google.com/youtube/terms/derived-metrics-policy) para o caso concreto. Acesso público não equivale a autorização irrestrita de armazenamento, agregação, download e treinamento.

O CSV público de SMTPD foi baixado e auditado, mas excluído da avaliação temporal por não fornecer datas de publicação. MicroLens foi considerado e não baixado: seus timestamps de interação não estabelecem o alvo comum após publicação. Veja a [auditoria de fontes](results/public/sources.json). Nenhum dos dois foi usado para alegar desempenho preditivo.

Não transfira rótulos de uma plataforma para outra. Para novos clientes ou anúncios, mantenha um perfil próprio de alvo/horizonte, atributos e validação.
