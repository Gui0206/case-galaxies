# Demonstração com modelos reais

**A ferramenta agora executa os modelos SMTPD salvos. Não treina nem coleta dados.** Ela estima visualizações em sete dias, mostra a referência histórica e a incerteza, e pode ordenar criativos do mesmo canal. Continua experimental: não houve ganho comprovado no teste reservado.

## Rode a demonstração

Na pasta do projeto, com o ambiente e os artefatos locais já existentes:

```bash
.venv/bin/python scripts/demo_real.py
```

O comando carrega os modelos, refaz três previsões e grava os detalhes em `.runtime/real-demo.json`. Não consulta a previsão salva para gerar sua resposta. Os três exemplos são os primeiros do arquivo publicado, em ordem temporal; não foram escolhidos pela qualidade do acerto.

| Vídeo | Histórico do canal | Previsão do modelo | Intervalo nominal de 90% | Resultado observado |
|---|---:|---:|---:|---:|
| a4TPwpWGN2I | 80.077 | 56.389 | 11.843–268.464 | 115.433 |
| 4f16xkWakkI | 35.632 | 47.723 | 10.023–227.206 | 26.342 |
| 6_ZP3BPzZxI | 39.903 | 38.197 | 8.022–181.856 | 43.328 |

**Nestes três exemplos, o histórico ficou mais perto do resultado real.** Eles demonstram o funcionamento, não constituem outra avaliação. No teste completo já encerrado, a redução do erro foi de apenas 0,18%, sem confirmação estatística. [Resultado completo e limites](SMTPD_RESULTS.md).

O primeiro exemplo usa atributos de edição extraídos do vídeo. Os outros dois usam a alternativa textual congelada, porque não têm mídia utilizável nessa execução. O resultado observado é anexado **depois** da inferência. A saída também inclui título/transcrição quando disponíveis, medidas de edição, cinco referências históricas recentes e contribuições calculadas pelo próprio modelo.

## Use uma entrada diferente

```bash
.venv/bin/python scripts/demo_real.py --request examples/real-creative.json
```

Esse arquivo é um cenário ilustrativo com um título novo e um canal presente na base; não possui resultado observado. Edite a legenda, duração e data. O modelo de linguagem local calcula a representação de um título novo; não há chamada a LLM, download ou treino.

| Campo | Significado |
|---|---|
| `creative_id` | Identificador do candidato; seu próprio resultado é excluído do histórico |
| `channel_id` | Canal com pelo menos três resultados maduros na base local |
| `caption` | Título/legenda que o público verá |
| `planned_datetime` | Data com fuso horário, a partir de 05/01/2024 |
| `duration_seconds` | Duração real em segundos |
| `category` | `2`: Autos & Vehicles; `26`: Howto & Style |
| `cached_video_id` | Opcional: reutiliza atributos de um vídeo já extraído |
| `media_path` | Opcional, em lugar do cache: caminho de vídeo local dentro de `.runtime` |

Sem mídia, a ferramenta usa o modelo textual congelado e avisa. Para arquivo local novo, o extrator existente pode fazer inferência de imagem, fala e edição, com os pesos já instalados; aceita até 180 segundos e 150 MB. Essa operação consome processamento, mas não treina nem coleta. **A demonstração padrão só reutiliza extrações existentes.**

## Python, API e MCP

Todos usam o mesmo adaptador:

```python
from pathlib import Path
from galaxies.real_service import RealCreativeRequest, predict_real

request = RealCreativeRequest.model_validate_json(
    Path("examples/real-creative.json").read_text()
)
result = predict_real(request)
print(result["explanation"])
```

Com o servidor existente (`.venv/bin/galaxies serve`), os novos endpoints são:

- `POST /api/real/predict`: um objeto como o arquivo de exemplo.
- `POST /api/real/compare`: `{"creatives": [objeto_a, objeto_b]}`; mesmo canal, categoria e horário, IDs distintos.
- `GET /api/real/replay/a4TPwpWGN2I`: refaz um exemplo e anexa seu resultado observado.

O servidor MCP (`.venv/bin/galaxies-mcp`) expõe `predict_real_creative`, `compare_real_creatives` e `replay_real_creative`. Os dois primeiros recebem `request`; o terceiro recebe `creative_id`. Os contratos completos aparecem no catálogo MCP e em `/docs` da API. A configuração de autenticação existente também protege os endpoints reais.

A interface visual antiga e as funções sem `real` continuam ligadas ao registro de demonstração sintética. Para usar os modelos reais, use os comandos e ferramentas desta página. Nenhum modelo foi promovido para produção.

## Como interpretar a resposta

- **Previsão:** número de views em sete dias; não prevê likes nem receita.
- **Histórico:** transformação inversa da média de `log(1 + views)` dos últimos até 20 resultados disponíveis. Não é a média aritmética das views.
- **Contexto:** referência adicional com duração, horário e histórico, sem a correção por conteúdo.
- **Intervalo:** nominalmente 90%; cobriu 88,93% no teste reservado. É amplo e não oferece garantia para campanhas atuais.
- **Ordenação:** usa o modelo de ranking congelado, diferente do modelo numérico. Por isso, não é necessariamente a ordem das estimativas de views. Não é probabilidade de viralizar. `recommended_id` permanece vazio.
- **Explicação:** contribuições SHAP e lineares do cálculo em escala logarítmica. Descrevem associações; não justificam promessas como “aumentar os cortes aumenta as views”. Componentes semânticos não têm tradução causal confiável.
- **Imagem e fala:** os extratores reais estão disponíveis e a transcrição aparece quando existe. A correção multimodal escolhida no desenvolvimento usa **apenas edição**. Não apresentamos imagem ou fala como fatores que melhoraram a previsão.

## Limites que permanecem

A base é pública, selecionada, de 2023–2024, com histórico incompleto. Howto & Style não equivale a uma validação específica para moda e beleza. Um canal desconhecido é recusado; uma data atual não cria histórico atual. O programa avisa sobre essa defasagem e sempre retorna confiança baixa e `eligible_for_decision=false`.

Os modelos e caches grandes permanecem locais, fora do Git. O manifesto versionado em `docs/results/smtpd/inference-manifest.json` registra seus hashes; o carregador verifica os arquivos antes do uso e verifica cada extração usada. Em uma cópia com esses mesmos artefatos, `scripts/prepare_real_service.py` prepara o manifesto local, sem produzir modelos. Clonar apenas o repositório não basta para rodar esta demonstração.

A verificação de integração reproduz seis previsões publicadas, com e sem mídia, verifica que resultados próprios/futuros não alteram a previsão e exercita API, MCP e comparação. Os testes de inferência bloqueiam treino CatBoost, ajuste do encoder e conexões de rede. Isso comprova a conexão da ferramenta; **não acrescenta evidência de ganho preditivo**.
