/* No external UI dependencies. All factual values below come from the local API. */
'use strict';
const $ = (selector, root = document) => root.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (value, digits = 0) => value == null ? '—' : new Intl.NumberFormat('pt-BR', {maximumFractionDigits:digits}).format(value);
const pct = value => value == null ? '—' : num(value * 100, 1) + '%';
const state = {models:[], datasets:[], active:null, report:null, scenario:null, comparison:null, uploads:{}, view:'overview', selection:null};
const viewNames = {overview:'Visão geral',compare:'Comparar criativos',models:'Modelos e evidências',data:'Dados e treinamento',guide:'Como funciona'};
let toastTimer, uploadIndex = 0, routeGeneration = 0;

async function api(path, options = {}) {
  const headers = {...options.headers};
  const token = sessionStorage.getItem('galaxies_token');
  if (token) headers.Authorization = `Bearer ${token}`;
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, {...options,headers});
  const result = await response.json();
  if (!response.ok) {
    if (response.status === 401) $('#token-dialog').showModal();
    const detail = result.detail;
    throw new Error(Array.isArray(detail) ? detail.map(d => `${d.loc?.join('.')}: ${d.msg}`).join('; ') : detail || 'Não foi possível concluir a operação.');
  }
  return result;
}
function toast(message, error = false) {
  const el = $('#toast'); el.textContent = message; el.className = error ? 'error' : ''; el.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => {el.hidden = true;}, error ? 9000 : 4500);
}
function badge(report = state.report) {
  return report?.evidence_level === 'synthetic_demo' ? '<span class="badge demo">◌ Demonstração sintética</span>' : report ? '<span class="badge green">Avaliação retrospectiva</span>' : '<span class="badge neutral">Sem modelo ativo</span>';
}
function disclaimer(report = state.report) {
  return report?.evidence_level === 'synthetic_demo' ? '<div class="notice"><strong>Ambiente de demonstração.</strong> Dados e cenários sintéticos, gerados para testar o fluxo. As métricas abaixo não representam acurácia em campanhas reais.</div>' : '<div class="notice info">Previsões são estimativas condicionadas aos dados disponíveis. Associações observadas não demonstram efeitos causais nem retorno de mídia paga.</div>';
}
function pageHead(title, subtitle, action = '') {
  return `<div class="page-head"><div><div class="eyebrow">CREATIVE INTELLIGENCE</div><h1>${title}</h1><p>${subtitle}</p></div>${action}</div>`;
}
function stat(label, value, note) { return `<div class="stat"><div class="stat-top">${label}<span>↗</span></div><div class="stat-value">${value}</div><div class="stat-note">${note}</div></div>`; }
function empty() { return '<div class="empty"><h2>Seu primeiro modelo começa aqui.</h2><p>Gere uma demonstração completa no terminal ou importe uma base autorizada.</p><pre>uv run galaxies demo</pre><a class="button primary" href="#data">Explorar dados e treinamento →</a></div>'; }
function pipeline() {
  return '<div class="steps">' + [['01','Criativo','Vídeo, áudio e texto'],['02','Atributos','Extração verificável'],['03','Previsão','Modelo estatístico'],['04','Decisão','Ranking e evidências']].map(([n,t,d]) => `<div class="step"><div class="step-num">${n}</div><h3>${t}</h3><p>${d}</p></div>`).join('') + '</div>';
}
async function refresh() {
  const [status, models, datasets] = await Promise.all([api('/api/status'),api('/api/models'),api('/api/datasets')]);
  const changedModel = state.active !== status.active_model_id;
  state.active = status.active_model_id; state.models = models.items; state.datasets = datasets.items;
  state.report = state.models.find(m => m.model_id === state.active) || null;
  $('#connection-text').textContent = 'Workspace local'; $('#connection-dot').className = 'online';
  if (!state.scenario || changedModel) await resetScenario();
}
async function resetScenario() {
  state.scenario = await api('/api/demo/scenario');
  Object.values(state.uploads).forEach(v=>URL.revokeObjectURL(v.preview));
  state.uploads={};state.comparison=null;
  if (!state.report) return;
  const ctx=state.scenario.context, r=state.report;
  ctx.metric=r.metric;ctx.horizon_days=r.horizon_days;
  if (r.evidence_level!=='synthetic_demo') {
    ctx.as_of=new Date().toISOString();ctx.planned_datetime=new Date(Date.now()+86400000).toISOString();
    ctx.channel_id='';ctx.subscribers=null;ctx.channel_stats_observed_at=null;
    ctx.sector=r.sectors[0];ctx.language=r.languages[0];ctx.distribution=r.distribution[0];
  }
}
function overview() {
  const r = state.report, m = r?.metrics;
  return pageHead('Da intuição à evidência.','Explore, compare e entenda o potencial dos seus criativos.',badge()) +
    `<section class="hero"><div><div class="eyebrow">ANTES DA PUBLICAÇÃO</div><h2>O próximo criativo.<br>Uma decisão mais informada.</h2><p>Compare alternativas com previsões estatísticas, intervalos de incerteza e evidências que você pode inspecionar.</p><a class="button primary" href="#compare">Comparar criativos <span>↗</span></a></div><div class="hero-art" aria-hidden="true"><div class="hero-orbit"></div><div class="hero-orbit second"></div><div class="orbit-core">✳</div><div class="float-card top"><span class="orb-icon">⌁</span><span><strong>Conteúdo + contexto</strong>Uma leitura em múltiplas camadas</span></div><div class="float-card bottom"><span class="orb-icon">◈</span><span><strong>Evidência, à vista</strong>Da extração à explicação</span></div></div></section>` +
    (r ? `<div class="stat-grid">${stat('CRIATIVOS NA BASE',num(r.audit.rows),`${num(r.audit.channels)} canais · ${Object.keys(r.audit.sectors).length} setores`)}${stat('ACERTO ENTRE PARES',pct(m.pair_accuracy),'Pares comparáveis no conjunto de teste')}${stat('COBERTURA DO INTERVALO',pct(m.interval_coverage),`Cobertura nominal: ${pct(r.config.coverage)}`)}${stat('HORIZONTE DE PREVISÃO',r.horizon_days+' dias',r.metric === 'likes' ? 'Curtidas acumuladas' : 'Visualizações acumuladas')}</div>${disclaimer(r)}` : empty()) +
    `<div class="grid-2"><section class="panel"><div class="panel-head"><div><h2>Um fluxo que você consegue acompanhar</h2><p>Cada etapa deixa um rastro verificável.</p></div><a href="#guide" class="arrow-link">Entenda o fluxo ↗</a></div>${pipeline()}</section><section class="panel"><div class="panel-head"><h2>Modelo em uso</h2><a href="#models" class="arrow-link">Ver evidências ↗</a></div><div class="model-summary"><div class="data-row"><span>Arquitetura</span><strong>CatBoost + calibração</strong></div><div class="data-row"><span>Validação</span><strong>Temporal, com expurgo</strong></div><div class="data-row"><span>Fonte dos números</span><strong>${r?.evidence_level === 'synthetic_demo' ? 'Dados sintéticos' : r ? 'Base importada' : 'Ainda não treinado'}</strong></div><div class="data-row"><span>Integração</span><strong>Python · API · MCP</strong></div></div></section></div>`;
}
const titles = ['Demonstração em primeiro plano','Produto e apresentação','Identidade da marca'];
function creativeCards() {
  return state.scenario.creatives.map((c,i) => {
    const uploaded = state.uploads[i];
    const hook = c.features.hook_clarity, benefit = c.features.benefit_clarity;
    return `<article class="creative-card"><div class="creative-cover ${['a','b','c'][i]}">${uploaded?.preview ? `<video src="${esc(uploaded.preview)}" controls preload="metadata" class="creative-video" aria-label="Prévia do criativo ${i+1}"></video>` : '<div class="product-art" aria-hidden="true">STUDIO</div>'}<span class="creative-letter">CRIATIVO ${String.fromCharCode(65+i)}</span><span class="creative-time">${num(c.features.duration_s)}s</span></div><div class="creative-body"><h3>${uploaded ? 'Seu vídeo' : titles[i]}</h3><p class="small muted">${uploaded ? 'Atributos extraídos do arquivo' : 'Cenário ilustrativo · atributos simulados'}</p><label>Legenda planejada<textarea data-caption="${i}">${esc(c.caption)}</textarea></label><div class="creative-foot"><button class="button secondary tiny" data-action="upload-video" data-index="${i}">↑ Enviar vídeo</button><span class="small muted upload-label">${esc(uploaded?.name || 'MP4 · MOV · WebM')}</span></div><div class="slider-row"><label>Abertura <span id="hook-value-${i}">${hook ?? '—'}</span><input type="range" min="0" max="10" step="1" value="${hook ?? 5}" data-slider="hook_clarity" data-index="${i}" ${uploaded ? 'disabled' : ''} aria-label="Clareza da abertura, criativo ${i+1}"></label><label>Benefício <span id="benefit-value-${i}">${benefit ?? '—'}</span><input type="range" min="0" max="10" step="1" value="${benefit ?? 5}" data-slider="benefit_clarity" data-index="${i}" ${uploaded ? 'disabled' : ''} aria-label="Clareza do benefício, criativo ${i+1}"></label></div></div></article>`;
  }).join('');
}
function compareView() {
  if (!state.report) return pageHead('Compare antes de publicar.','Uma leitura conjunta de conteúdo, contexto e incerteza.') + empty();
  const ctx = state.scenario.context;
  return pageHead('Compare antes de publicar.','Mesmo canal. Mesmo contexto. Alternativas lado a lado.', '<button class="button secondary" data-action="reset">↻ Restaurar cenário</button>') + disclaimer() +
    `<section class="panel context-panel"><div class="panel-head"><h2>01 · Contexto da publicação</h2><span class="badge neutral">${ctx.horizon_days} dias · ${ctx.metric === 'likes' ? 'curtidas' : 'views'}</span></div><div class="form-grid"><label>Canal<input id="channel" value="${esc(ctx.channel_id)}" placeholder="Identificador do canal"></label><label>Setor<select id="sector">${['beleza','automotivo','moda'].map(s => `<option ${s===ctx.sector?'selected':''} value="${s}">${s[0].toUpperCase()+s.slice(1)}</option>`).join('')}</select></label><label>Publicação planejada <span class="muted">(UTC)</span><input id="planned" type="datetime-local" value="${esc(ctx.planned_datetime.slice(0,16))}"></label></div><p class="form-note">O cenário padrão usa datas históricas de demonstração. Ao alterar a data, a decisão passa a usar o momento atual.</p></section>` +
    `<div class="panel-head"><h2>02 · Seus criativos</h2><span class="small muted">3 alternativas · você pode editar as legendas</span></div><div class="creative-grid">${creativeCards()}</div><div class="run-bar"><div><h3>Pronto para comparar?</h3><p>O modelo calcula as estimativas e indica quando a evidência é insuficiente.</p></div><button class="button primary" id="compare-button" data-action="compare">Analisar os 3 criativos ↗</button></div><div id="comparison-results">${state.comparison ? renderResults(state.comparison) : ''}</div>`;
}
function intervalSvg(p, max) {
  const scale = x => 8 + x / max * 214;
  return `<svg class="range-svg" viewBox="0 0 230 50" role="img" aria-label="Intervalo de ${num(p.interval_low)} a ${num(p.interval_high)}; previsão ${num(p.predicted_metric)}"><line x1="8" y1="18" x2="222" y2="18" stroke="#edf0e6" stroke-width="4"/><line x1="${scale(p.interval_low)}" y1="18" x2="${scale(p.interval_high)}" y2="18" stroke="#adc8aa" stroke-width="5" stroke-linecap="round"/><circle cx="${scale(p.predicted_metric)}" cy="18" r="4" fill="#406f48"/><text x="8" y="42" class="svg-label">${num(p.interval_low)}</text><text x="222" y="42" text-anchor="end" class="svg-label">${num(p.interval_high)}</text></svg>`;
}
function driversHtml(drivers) {
  const max = Math.max(...drivers.map(d=>Math.abs(d.contribution_log)),.01);
  return drivers.map(d => `<div class="driver"><span>${esc(d.label)}</span><div class="driver-track"><i class="${d.direction==='negative'?'negative':''}" style="width:${Math.abs(d.contribution_log)/max*100}%"></i></div><span>${d.contribution_log>=0?'+':''}${num(d.contribution_log,2)}</span></div>`).join('');
}
function renderResults(result) {
  const max = Math.max(...result.ranking.map(p=>p.interval_high),1), best = result.ranking[0];
  const decisionLabels = {demonstration_only:'Ranking ilustrativo',insufficient_evidence:'Evidência insuficiente',technical_tie:'Empate técnico',recommendation:'Comparação concluída'};
  return `<section class="results"><div class="result-head"><div><div class="eyebrow">RESULTADO DA ANÁLISE</div><h2>${decisionLabels[result.decision]}</h2></div><button class="button secondary tiny" data-action="download">↓ Exportar análise</button></div><div class="notice ${result.decision==='recommendation'?'info':''}">${esc(result.reason)}</div>` +
    result.ranking.map((p,i) => `<div class="rank-card"><div class="rank-num">${i+1}</div><div><div class="rank-title">${esc(p.creative_id.replace('criativo-','Criativo ').toUpperCase())}</div><div class="rank-sub">${p.relative_to_channel == null ? 'Sem histórico suficiente' : num(p.relative_to_channel,2)+'× a mediana do canal'}</div></div><div><div class="rank-value">${num(p.predicted_metric)}</div><div class="rank-sub">${p.metric === 'likes' ? 'curtidas estimadas' : 'views estimadas'}</div></div>${intervalSvg(p,max)}<button class="button secondary tiny" data-action="detail" data-id="${esc(p.creative_id)}">Evidências ↗</button></div>`).join('') +
    `<div class="insights-grid"><section class="panel"><div class="panel-head"><div><h2>O que influenciou a primeira previsão</h2><p>Contribuições SHAP em escala log</p></div></div>${driversHtml(best.top_drivers)}<p class="form-note">Associações aprendidas pelo modelo. Alterar um atributo não garante um ganho equivalente.</p></section><section class="panel"><div class="panel-head"><div><h2>Comparações entre pares</h2><p>Probabilidade calibrada em pares históricos</p></div></div>${result.pairwise.map(p=>`<div class="probability-row"><span>${esc(p.a.replace('criativo-',''))} superar ${esc(p.b.replace('criativo-',''))}</span><strong>${pct(p.probability_a_better)}</strong></div>`).join('')}<p class="form-note">${result.evidence_level==='synthetic_demo'?'Probabilidades do cenário sintético; sem validade para campanhas reais.':'Pares do mesmo canal e janela de tempo. A validade em campanhas requer avaliação própria.'}</p></section></div></section>`;
}
function metricBars(r) {
  return `<div class="metric-bars">${[['Modelo multimodal',r.metrics.pair_accuracy,false],['Histórico do canal',r.baseline.pair_accuracy,true]].map(([label,v,b])=>`<div class="bar-row"><span>${label}</span><div class="bar-track"><i class="${b?'baseline':''}" style="width:${(v||0)*100}%"></i></div><strong>${pct(v)}</strong></div>`).join('')}</div>`;
}
function scatter(items) {
  const max = Math.max(...items.flatMap(p=>[Math.log1p(p.actual),Math.log1p(p.prediction)]),1);
  const x = v=>42+Math.log1p(v)/max*315, y=v=>208-Math.log1p(v)/max*180;
  return `<svg class="scatter" viewBox="0 0 390 245" role="img" aria-label="Dispersão das previsões versus resultados observados, em escala log"><line x1="42" y1="208" x2="357" y2="28" stroke="#d6dfcf" stroke-dasharray="5 5"/><line x1="42" y1="208" x2="363" y2="208" stroke="#dce3d6"/><line x1="42" y1="24" x2="42" y2="208" stroke="#dce3d6"/>${items.slice(0,400).map(p=>`<circle cx="${x(p.actual)}" cy="${y(p.prediction)}" r="2.7" fill="#7ca887" opacity=".45"><title>Real: ${num(p.actual)} · previsão: ${num(p.prediction)}</title></circle>`).join('')}<text x="190" y="238" class="svg-label" text-anchor="middle">Resultado observado (log)</text><text x="9" y="16" class="svg-label">Previsão (log)</text><text x="34" y="222" class="svg-label">0</text></svg>`;
}
async function modelsView() {
  const r = state.models.find(m => m.model_id === state.selection) || state.report || state.models[0];
  if (!r) return pageHead('A evidência por trás da previsão.','Modelos, validação e limitações em um só lugar.') + empty();
  const predictions = await api(`/api/models/${r.model_id}/predictions`);
  return pageHead('A evidência por trás da previsão.','Resultados de teste, referências simples e incerteza.',badge(r)) +
    `<label>Versão do modelo<select id="model-select">${state.models.map(m=>`<option value="${esc(m.model_id)}" ${m.model_id===r.model_id?'selected':''}>${esc(m.model_id)}${m.model_id===state.active?' · ativo':''}${m.evidence_level==='synthetic_demo'?' · sintético':''}</option>`).join('')}</select></label>${disclaimer(r)}` +
    `<div class="stat-grid">${stat('ACERTO ENTRE PARES',pct(r.metrics.pair_accuracy),num(r.metrics.pair_count)+' pares históricos')}${stat('ERRO ABSOLUTO EM LOG',num(r.metrics.mae_log,3),'Menor é melhor')}${stat('CORRELAÇÃO NO CANAL',num(r.metrics.spearman_within_channel,3),'Spearman médio por canal')}${stat('COBERTURA OBSERVADA',pct(r.metrics.interval_coverage),num(r.metrics.rows)+' criativos de teste')}</div><div class="grid-2"><section class="panel"><div class="panel-head"><div><h2>O modelo supera a referência?</h2><p>Acerto de ordenação entre pares comparáveis</p></div></div>${metricBars(r)}<div class="notice info">Diferença média por canal: ${num(r.paired_channel_bootstrap.mean_delta == null ? null : r.paired_channel_bootstrap.mean_delta * 100,1)} p.p. Intervalo de 95%: ${num(r.paired_channel_bootstrap.ci95_low == null ? null : r.paired_channel_bootstrap.ci95_low * 100,1)} a ${num(r.paired_channel_bootstrap.ci95_high == null ? null : r.paired_channel_bootstrap.ci95_high * 100,1)} p.p.</div><p class="form-note">Bootstrap por canal. Pares do mesmo vídeo não são observações independentes.</p></section><section class="panel"><div class="panel-head"><div><h2>Previsão × resultado</h2><p>A linha diagonal representa concordância perfeita.</p></div></div>${scatter(predictions.items)}</section></div>` +
    `<section class="panel full"><div class="panel-head"><div><h2>Uma validação que respeita o tempo</h2><p>Dados de treino, seleção, calibração e teste em períodos distintos.</p></div><span class="badge neutral">${num(r.purged_rows)} registros expurgados</span></div><div class="timeline">${[['train','Treino',55],['validation','Validação',15],['calibration','Calibração',15],['test','Teste',15]].map(([key,name,w])=>`<span style="flex:${w}">${name} · ${num(r.splits[key])}</span>`).join('')}</div><p class="form-note">As faixas mostram proporções planejadas; as contagens refletem expurgos. Resultados que amadurecem após o corte não entram na etapa anterior.</p></section>` +
    `<section class="panel full"><div class="panel-head"><h2>Desempenho por setor</h2></div><div class="table-scroll"><table><thead><tr><th>Setor</th><th>Criativos</th><th>Acerto em pares</th><th>Erro em log</th><th>Cobertura</th></tr></thead><tbody>${Object.entries(r.by_sector).map(([sector,m])=>`<tr><td>${esc(sector)}</td><td>${num(m.rows)}</td><td>${pct(m.pair_accuracy)}</td><td>${num(m.mae_log,3)}</td><td>${pct(m.interval_coverage)}</td></tr>`).join('')}</tbody></table></div></section><section class="panel full"><div class="panel-head"><h2>Limites desta versão</h2>${r.model_id!==state.active?`<button class="button secondary tiny" data-action="promote" data-model="${esc(r.model_id)}">Ativar ${r.evidence_level==='synthetic_demo'?'demonstração':'modelo'}</button>`:'<span class="badge green">Modelo ativo</span>'}</div><div class="prose">${r.limitations.map(l=>`<p>• ${esc(l)}</p>`).join('')}</div><pre>${esc(JSON.stringify({model_id:r.model_id,evidence_level:r.evidence_level,eligible_for_decision:r.eligible_for_decision,dataset_sha256:r.dataset_sha256,environment:r.environment},null,2))}</pre></section>`;
}
async function dataView() {
  const jobs = await api('/api/jobs');
  return pageHead('Dados bons. Decisões sustentáveis.','Importe uma base, confira o contrato e acompanhe o treinamento.','<button class="button primary" data-action="upload-dataset">↑ Importar base</button>') +
    `<div class="grid-2"><section class="panel"><div class="panel-head"><div><h2>Bases disponíveis</h2><p>JSONL, JSON ou Parquet · contrato validado</p></div><span class="badge neutral">${state.datasets.length} bases</span></div>${state.datasets.length ? `<table><thead><tr><th>Arquivo</th><th>Tamanho</th><th></th></tr></thead><tbody>${state.datasets.map(d=>`<tr><td class="filename">${esc(d.name)}</td><td>${num(d.bytes/1024)} KB</td><td><button class="button secondary tiny" data-action="audit" data-name="${esc(d.name)}">Inspecionar</button></td></tr>`).join('')}</tbody></table>` : '<p class="prose">Nenhuma base importada. Use uma base autorizada no contrato documentado ou gere a demonstração pelo terminal.</p>'}</section><section class="panel"><div class="panel-head"><h2>Treinar uma nova versão</h2></div><label>Base de treinamento<select id="train-dataset">${state.datasets.map(d=>`<option>${esc(d.name)}</option>`).join('')}</select></label><label class="section-space">Iterações máximas<input id="train-iterations" type="number" min="20" max="2000" value="240"></label><p class="form-note">O treino usa parada antecipada, separação temporal e calibração. Cada execução cria uma versão; a ativação é uma ação separada.</p><button class="button primary section-space" data-action="train" id="train-button" ${!state.datasets.length?'disabled':''}>Iniciar treinamento →</button></section></div><section class="panel full"><div class="panel-head"><div><h2>Execuções recentes</h2><p>Fila local persistente · uma execução de treino por vez</p></div><button class="button secondary tiny" data-action="refresh">↻ Atualizar</button></div><div id="job-list">${renderJobs(jobs.items)}</div></section><section class="panel full"><div class="panel-head"><h2>Antes de importar</h2><a class="arrow-link" href="#guide">Ver contrato e metodologia ↗</a></div><div class="prose"><p><strong>Uma linha, um criativo e um horizonte.</strong> Declare publicação, disponibilidade dos atributos, observação da métrica, origem e base de uso. A medição precisa ocorrer no horizonte escolhido, com tolerância máxima de 24 horas.</p><p><strong>Dados sintéticos e reais permanecem separados.</strong> O sistema recusa mistura de origens, métricas, horizontes e estatísticas futuras do canal.</p></div></section>`;
}
function renderJobs(jobs) {
  const labels = {queued:'Na fila',running:'Treinando',completed:'Concluído',failed:'Falhou',interrupted:'Interrompido'};
  return jobs.length ? jobs.map(j=>`<div class="job-item"><div><strong>${esc(j.payload.dataset)}</strong><div class="small muted model-id">${esc(j.id.slice(0,12))} · ${esc(new Date(j.created_at).toLocaleString('pt-BR'))}</div>${j.error?`<p class="small">${esc(j.error)}</p>`:''}</div><span class="badge ${j.status==='completed'?'green':'neutral'}">${labels[j.status]}</span>${j.result?`<button class="button secondary tiny" data-action="open-model" data-model="${esc(j.result.model_id)}">Ver modelo</button>`:['failed','interrupted'].includes(j.status)?`<button class="button secondary tiny" data-action="retry" data-job="${esc(j.id)}">Reiniciar</button>`:''}</div>`).join('') : '<p class="prose">Nenhum treinamento solicitado pela API. Modelos gerados pelo terminal aparecem na área de modelos.</p>';
}
async function guideView() {
  const prospective = await api('/api/prospective');
  return pageHead('Entender faz parte da decisão.','Um guia visual do criativo até a evidência.','<a class="button secondary" href="/docs" target="_blank" rel="noreferrer">API interativa ↗</a>') +
    `<div class="guide-flow">${[['01','Observe o conteúdo','Frames, transcrição, áudio e atributos estruturados descrevem o criativo.'],['02','Considere o contexto','O histórico usa somente resultados conhecidos no instante da decisão.'],['03','Estime com um modelo','CatBoost calcula as previsões. Quantis e calibração produzem os intervalos.'],['04','Inspecione a evidência','Ranking, SHAP, comparáveis e limitações ficam disponíveis em uma saída estruturada.']].map(([n,t,d])=>`<article class="guide-card"><div class="step-num">${n}</div><h3>${t}</h3><p>${d}</p></article>`).join('')}</div><div class="grid-2"><section class="panel"><div class="panel-head"><h2>Como ler uma previsão</h2></div><div class="prose"><p><strong>Estimativa central:</strong> o número previsto pelo modelo para o horizonte escolhido.</p><p><strong>Intervalo de predição:</strong> a faixa produzida pelo modelo e calibrada em um período separado. A cobertura precisa continuar sendo monitorada.</p><p><strong>Desempenho relativo:</strong> a previsão dividida pela mediana histórica disponível do canal. Não representa ganho causal.</p><p><strong>Empate técnico:</strong> a comparação não atingiu o limiar operacional para recomendar um vencedor.</p><p><strong>Evidência insuficiente:</strong> dados ausentes, entrada fora do domínio ou modelo sem validação adequada.</p></div></section><section class="panel"><div class="panel-head"><h2>Três formas de integrar</h2></div><div class="prose"><p><strong>Python:</strong> importe <code>galaxies.service.predict</code> e passe os contratos tipados.</p><p><strong>HTTP:</strong> envie <code>POST /api/predict</code> ou <code>POST /api/compare</code>.</p><p><strong>MCP:</strong> execute <code>uv run galaxies mcp</code>. A persona pode consultar a ferramenta e receber a mesma saída estruturada.</p></div><pre>creative + context\n        ↓\nprediction + interval + ranking\n        ↓\ndrivers + comparables + limitations</pre></section></div><section class="panel full"><div class="panel-head"><div><h2>Validação prospectiva</h2><p>Previsões registradas antes da publicação e resultados adicionados depois.</p></div><span class="badge neutral">${prospective.registered} registros</span></div><div class="stat-grid">${stat('PREVISÕES REGISTRADAS',num(prospective.registered),'Registro imutável')}${stat('RESULTADOS OBSERVADOS',num(prospective.matured),'No horizonte definido')}${stat('ERRO EM LOG',num(prospective.mae_log,3),'Somente resultados recebidos')}${stat('COBERTURA',pct(prospective.coverage),'Somente resultados recebidos')}</div><p class="prose">Esse registro aceita modelos reais e publicações futuras. A demonstração sintética não pode produzir evidência prospectiva real. Consulte os endpoints de <code>/api/prospective</code> na documentação.</p></section><section class="panel full"><div class="panel-head"><h2>Fluxo rápido pelo terminal</h2></div><pre>uv sync --group dev\nuv run galaxies demo\nuv run galaxies serve\n\n# Para uma base real autorizada:\nuv run galaxies validate minha-base.jsonl\nuv run galaxies import-data minha-base.jsonl\nuv run galaxies train minha-base.jsonl\n\n# Diagnóstico do ambiente e comandos:\nuv run galaxies doctor\nuv run galaxies --help</pre></section>`;
}
async function route() {
  const generation = ++routeGeneration;
  state.view = location.hash.slice(1) || 'overview';
  if (!viewNames[state.view]) state.view = 'overview';
  document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.dataset.view===state.view));
  $('#breadcrumb').innerHTML = `Workspace <span>/</span> ${viewNames[state.view]}`;
  $('#main').innerHTML = '<div class="loading-card"><span class="spinner"></span> Carregando…</div>';
  try {
    const views = {overview,compare:compareView,models:modelsView,data:dataView,guide:guideView};
    const html = await views[state.view]();
    if (generation === routeGeneration) { $('#main').innerHTML = html; window.scrollTo({top:0}); }
  } catch(error) {
    if (generation === routeGeneration) $('#main').innerHTML = `<div class="empty"><h2>Não foi possível carregar esta área.</h2><p>${esc(error.message)}</p><button class="button primary" data-action="refresh">Tentar novamente</button></div>`;
  }
}
function showDetail(prediction) {
  $('#details-content').innerHTML = `<div class="eyebrow">RASTRO DA ANÁLISE</div><h2>${esc(prediction.creative_id)}</h2><p>${esc(prediction.explanation)}</p>${prediction.confidence_reasons.map(r=>`<div class="notice">${esc(r)}</div>`).join('')}<h3>Contribuições do modelo</h3>${driversHtml(prediction.top_drivers)}<h3 class="section-space">Criativos históricos comparáveis</h3>${prediction.comparables.map(c=>`<div class="data-row"><span>${esc(c.caption)}</span><strong>${num(c.observed_metric)}</strong></div>`).join('')||'<p>Sem exemplos elegíveis.</p>'}<p class="small">Os comparáveis são referências observacionais, não prova de causalidade.</p><h3>Evidências extraídas</h3>${prediction.extraction_evidence.map(e=>`<p>${num(e.start_seconds,1)}–${num(e.end_seconds,1)}s · ${esc(e.observation)}</p>`).join('')||'<p>Nenhuma evidência temporal extraída neste cenário simulado.</p>'}<p class="model-id">Execução: ${esc(prediction.trace_id)}</p><button class="button secondary" data-action="trace" data-trace="${esc(prediction.trace_id)}">Inspecionar saída completa ↗</button>`;
  $('#details-dialog').showModal();
}
async function pollJob(jobId) {
  for (;;) {
    await new Promise(resolve=>setTimeout(resolve,1500));
    const job = await api(`/api/jobs/${jobId}`);
    if (state.view === 'data' && $('#job-list')) $('#job-list').innerHTML = renderJobs((await api('/api/jobs')).items);
    if (['completed','failed','interrupted'].includes(job.status)) {
      if (job.status==='completed') {await refresh();toast('Treinamento concluído. A nova versão está disponível em Modelos.');}
      else toast(job.error || 'Treinamento interrompido.',true);
      return;
    }
  }
}
document.addEventListener('click', async event => {
  const button = event.target.closest('[data-action]'); if (!button) return;
  const action = button.dataset.action;
  try {
    if (action==='refresh') {await refresh(); await route();}
    if (action==='reset') {await resetScenario();await route();}
    if (action==='upload-video') {uploadIndex = Number(button.dataset.index);$('#video-upload').click();}
    if (action==='upload-dataset') $('#dataset-upload').click();
    if (action==='compare') {
      state.scenario.context.channel_id = $('#channel').value;
      state.scenario.context.sector = $('#sector').value;
      state.scenario.model_id = state.active;
      button.disabled = true;button.innerHTML='<span class="spinner"></span> Analisando…';
      state.comparison = await api('/api/compare',{method:'POST',body:state.scenario});
      $('#comparison-results').innerHTML = renderResults(state.comparison);
      $('#comparison-results').scrollIntoView({behavior:'smooth',block:'start'});
      button.disabled=false;button.textContent='Analisar novamente ↗';
    }
    if (action==='detail') showDetail(state.comparison.ranking.find(p=>p.creative_id===button.dataset.id));
    if (action==='download') {
      const url = URL.createObjectURL(new Blob([JSON.stringify(state.comparison,null,2)],{type:'application/json'}));
      const a=document.createElement('a');a.href=url;a.download='galaxies-analise.json';a.click();URL.revokeObjectURL(url);
    }
    if (action==='trace') {const data = await api(`/api/traces/${button.dataset.trace}`);$('#details-content').innerHTML='<h2>Execução completa</h2><pre>'+esc(JSON.stringify(data,null,2))+'</pre>';}
    if (action==='audit') {const data = await api(`/api/datasets/${encodeURIComponent(button.dataset.name)}/audit`);$('#details-content').innerHTML='<h2>Auditoria da base</h2><pre>'+esc(JSON.stringify(data,null,2))+'</pre>';$('#details-dialog').showModal();}
    if (action==='train') {
      button.disabled=true;
      const job=await api('/api/train',{method:'POST',body:{dataset:$('#train-dataset').value,config:{iterations:Number($('#train-iterations').value)}}});
      toast('Treinamento iniciado. Você pode acompanhar a execução nesta área.');
      await route();pollJob(job.id).catch(e=>toast(e.message,true));
    }
    if (action==='retry') {const job=await api(`/api/jobs/${button.dataset.job}/retry`,{method:'POST'});await route();pollJob(job.id).catch(e=>toast(e.message,true));}
    if (action==='open-model') {state.selection=button.dataset.model;location.hash='models';}
    if (action==='promote') {const selected=state.models.find(m=>m.model_id===button.dataset.model);await api(`/api/models/${selected.model_id}/promote`,{method:'POST',body:{demo:selected.evidence_level==='synthetic_demo'}});await refresh();await route();toast('Modelo ativado.');}
  } catch(error) {toast(error.message,true);if(button.isConnected){button.disabled=false;if(action==='compare')button.textContent='Tentar análise novamente ↗';}}
});
document.addEventListener('input', event=>{
  const el=event.target;
  if(el.dataset.caption!==undefined) state.scenario.creatives[Number(el.dataset.caption)].caption=el.value;
  if(el.dataset.slider){const i=Number(el.dataset.index);state.scenario.creatives[i].features[el.dataset.slider]=Number(el.value);$(`#${el.dataset.slider==='hook_clarity'?'hook':'benefit'}-value-${i}`).textContent=el.value;}
});
document.addEventListener('change',async event=>{
  if(event.target.id==='model-select'){state.selection=event.target.value;await route();}
  if(event.target.id==='planned'){
    const value=event.target.value;if(value){state.scenario.context.planned_datetime=new Date(value+'Z').toISOString();state.scenario.context.as_of=new Date().toISOString();}
  }
});
$('#video-upload').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  const index=uploadIndex;
  try{
    toast('Enviando vídeo e extraindo atributos locais…');
    const form=new FormData();form.append('file',file);
    const upload=await api('/api/media',{method:'POST',body:form});
    const attrs=await api(`/api/media/${upload.media_id}/extract`,{method:'POST',body:{}});
    state.scenario.creatives[index]={creative_id:`criativo-${String.fromCharCode(97+index)}`,attributes_source:"media_extraction",caption:state.scenario.creatives[index].caption,transcript:attrs.transcript,features:attrs.features,visual_embedding:attrs.visual_embedding,evidence:attrs.evidence,media_id:upload.media_id};
    if(state.uploads[index])URL.revokeObjectURL(state.uploads[index].preview);
    state.uploads[index]={name:file.name,preview:URL.createObjectURL(file)};state.comparison=null;await route();
    toast('Vídeo processado. A extração básica deixa ausentes transcrição e rubrica semântica; habilite os módulos opcionais pelo CLI.');
  }catch(error){toast(error.message,true);}finally{event.target.value='';}
});
$('#dataset-upload').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  try{const form=new FormData();form.append('file',file);await api('/api/datasets',{method:'POST',body:form});await refresh();await route();toast('Base validada e importada.');}catch(error){toast(error.message,true);}finally{event.target.value='';}
});
$('.dialog-close').addEventListener('click',()=>$('#details-dialog').close());
$('#token-button').addEventListener('click',()=>$('#token-dialog').showModal());
$('#token-cancel').addEventListener('click',()=>$('#token-dialog').close());
$('#token-form').addEventListener('submit',async event=>{event.preventDefault();sessionStorage.setItem('galaxies_token',$('#token-input').value);$('#token-input').value='';$('#token-dialog').close();try{await refresh();await route();}catch(e){toast(e.message,true);}});
window.addEventListener('hashchange',route);
refresh().then(route).catch(error=>{ $('#connection-text').textContent='Sem conexão';$('#main').innerHTML=`<div class="empty"><h2>Vamos conectar o workspace.</h2><p>${esc(error.message)}</p><button class="button primary" data-action="refresh">Tentar novamente</button></div>`; });
