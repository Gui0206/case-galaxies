'use strict';
const $ = s => document.querySelector(s);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = v => new Intl.NumberFormat('pt-BR', {maximumFractionDigits:0}).format(v);
let options, preview, busy = false, mode = 'own';
async function api(path, init = {}) {
  const headers = {...init.headers};
  const token = sessionStorage.getItem('galaxies_token');
  if (token) headers.Authorization = `Bearer ${token}`;
  if (init.body && !(init.body instanceof FormData)) {headers['Content-Type']='application/json';init.body=JSON.stringify(init.body);}
  const response = await fetch(path, {...init, headers});
  let data; try {data = await response.json();} catch {throw new Error('O servidor não respondeu corretamente. Verifique se ele continua aberto.');}
  if (!response.ok) throw new Error(response.status===401 ? 'Informe o token em Configurar acesso.' : Array.isArray(data.detail) ? data.detail.map(d=>d.msg).join('; ') : data.detail || 'Falha no processamento local. Nenhuma previsão foi produzida.');
  return data;
}
function fail(error){$('#error').textContent=error.message;$('#error').hidden=false;$('#status').textContent='Não foi possível concluir.';}
function setBusy(value){busy=value;$('#inputs').disabled=value||!options;$('#example').disabled=value||!options;$('#connect').disabled=value;$('#own-mode').disabled=value;$('#demo-mode').disabled=value;$('#demo-video').disabled=value;}
function clearResult(){if(!busy){$('#result').hidden=true;$('#error').hidden=true;$('#status').textContent=mode==='own'?'Envie seu vídeo para obter uma descrição, sem previsão de views.':'Escolha uma publicação da base para comparar modelo, histórico e resultado observado.';}}
async function load(){try{options=await api('/api/real/options');$('#file-info').textContent=`Até ${options.max_upload_mb} MB. O arquivo fica neste computador.`;setBusy(false);clearResult();}catch(e){fail(e);}}
function chooseMode(next){if(busy)return;mode=next;$('#own-panel').hidden=mode!=='own';$('#demo-panel').hidden=mode!=='demo';$('#own-mode').setAttribute('aria-pressed',String(mode==='own'));$('#demo-mode').setAttribute('aria-pressed',String(mode==='demo'));$('#result-heading').textContent=mode==='own'?'2. Descrição e limites':'2. Previsão e evidência';clearResult();}
$('#own-mode').addEventListener('click',()=>chooseMode('own'));
$('#demo-mode').addEventListener('click',()=>chooseMode('demo'));
function renderAnalysis(r,title){
 const decimal=v=>v==null?'não disponível':new Intl.NumberFormat('pt-BR',{maximumFractionDigits:2}).format(v);
 $('#status').textContent='Análise concluída. Nenhuma previsão de views foi calculada.';
 $('#result').innerHTML=`<span class="tag">Descrição do arquivo · sem previsão de desempenho</span><h3>${esc(title)}</h3><div class="cards"><div class="card"><span>Duração</span><strong>${decimal(r.duration_seconds)} s</strong></div><div class="card"><span>Planos detectados</span><strong>${num(r.editing.shot_count)}</strong></div></div><h3>Edição e imagem</h3><table><tbody><tr><td>Cortes por segundo</td><td>${decimal(r.editing.cuts_per_second)}</td></tr><tr><td>Duração média dos planos</td><td>${decimal(r.editing.shot_mean_s)} s</td></tr><tr><td>Primeiro plano</td><td>${decimal(r.editing.first_shot_s)} s</td></tr><tr><td>Luminosidade média (0–1)</td><td>${decimal(r.editing.brightness_mean)}</td></tr><tr><td>Quadros usados no encoder visual</td><td>${num(r.sampled_frames)}</td></tr></tbody></table><p class="muted">Valores descritivos. Mais cortes ou luminosidade não significam conteúdo melhor. As representações visuais não geram rótulos confiáveis de produto ou qualidade.</p><h3>Fala reconhecida</h3><p>Idioma detectado: ${esc(r.language||'não identificado')}. A transcrição pode conter erros.</p><pre>${esc(r.transcript||'Nenhuma fala reconhecida.')}</pre><h3>O que esta análise não permite concluir</h3><ul>${r.limitations.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`;
 $('#result').hidden=false;
}
function render(r, title){
 $('#status').textContent='Concluído. Estimativa experimental, sem recomendação de investimento.';
 const observed=r.demonstration?.observed_views_7d;
 $('#result').innerHTML=`<span class="tag">Modelo real · confiança baixa</span><h3>${esc(title)}</h3><p class="muted">Canal ${esc(r.channel_id)} · contexto em ${esc(new Date(r.history_available_before).toLocaleString('pt-BR'))}</p>
 <div class="cards"><div class="card"><span>Previsão em 7 dias</span><strong>${num(r.predicted_metric)}</strong><span>visualizações</span></div><div class="card"><span>Referência do histórico</span><strong>${num(r.historical_baseline)}</strong><span>visualizações</span></div></div>
 <p class="interval"><strong>Intervalo: ${num(r.interval_low)} a ${num(r.interval_high)} views</strong><br>Nominalmente 90%; cobertura observada de 88,93% no teste de 2024. Não é garantia para hoje.</p>
 ${observed==null?'':`<p><strong>Resultado real deste exemplo: ${num(observed)} views.</strong> Anexado depois da previsão; não foi usado como entrada.</p>`}
 <p>${esc(r.explanation)}</p><h3>O que foi usado?</h3><p>${r.media_used_for_prediction?'Histórico, contexto, título e atributos de edição do vídeo.':'Histórico, contexto e título. A previsão não usou mídia.'} ${r.observed_content?'Imagem e fala também foram extraídas, mas não entram na correção multimodal selecionada.':''}</p>
 <h3>Limitações desta previsão</h3><ul>${r.confidence_reasons.map(reason=>`<li>${esc(reason)}</li>`).join('')}</ul>
 <details><summary>Ver o histórico usado como referência</summary><p>${num(r.history_count)} resultados anteriores disponíveis. A referência usa até 20; abaixo estão os cinco mais recentes. Não é uma média aritmética.</p><table><thead><tr><th>Publicação anterior</th><th>Views em 7 dias</th></tr></thead><tbody>${r.recent_history.map(h=>`<tr><td>${esc(h.caption)}<br><small>${esc(new Date(h.published_at).toLocaleDateString('pt-BR'))}</small></td><td>${num(h.views_7d)}</td></tr>`).join('')}</tbody></table></details>
 ${r.observed_content?`<details><summary>Ver análise do vídeo</summary><p>${num(r.observed_content.duration_seconds)} segundos · ${r.observed_content.semantic_frames} quadros analisados.</p><p>Medidas de edição descrevem o vídeo; não provam que mais cortes causam mais views.</p><h3>Fala reconhecida</h3><pre>${esc(r.observed_content.transcript||'Nenhuma fala reconhecida.')}</pre></details>`:''}`;
 $('#result').hidden=false;
}
$('#predict-form').addEventListener('submit',async event=>{
 event.preventDefault();if(busy)return;clearResult();setBusy(true);
 try{
  const file=$('#video').files[0];if(!file)throw new Error('Selecione um vídeo.');
  if(file.size>options.max_upload_mb*1024*1024)throw new Error(`O arquivo excede ${options.max_upload_mb} MB.`);
  $('#status').textContent='1 de 2 · Salvando e verificando o vídeo local…';
  const body=new FormData();body.append('file',file);
  const upload=await api('/api/media',{method:'POST',body});
  $('#status').textContent='2 de 2 · Descrevendo imagem, fala e edição com os pesos locais. Pode levar alguns minutos. Não estamos calculando views.';
  const result=await api(`/api/real/media/${upload.media_id}/analyze`,{method:'POST'});
  renderAnalysis(result,file.name);
 }catch(e){fail(e);}finally{setBusy(false);}
});
$('#example').addEventListener('click',async()=>{if(busy)return;clearResult();setBusy(true);$('#status').textContent='Recalculando o exemplo com os modelos reais e a extração já salva…';try{render(await api(`/api/real/replay/${$('#demo-video').value}`),`Exemplo histórico · ${$('#demo-video').value}`);}catch(e){fail(e);}finally{setBusy(false);}});
$('#video').addEventListener('change',()=>{if(preview)URL.revokeObjectURL(preview);const file=$('#video').files[0];$('#preview').hidden=!file;if(file){preview=URL.createObjectURL(file);$('#preview').src=preview;}else{$('#preview').removeAttribute('src');}});
$('#demo-video').addEventListener('change',clearResult);
$('#predict-form').addEventListener('input',clearResult);
$('#connect').addEventListener('click',()=>{sessionStorage.setItem('galaxies_token',$('#token').value);load();});
load();
