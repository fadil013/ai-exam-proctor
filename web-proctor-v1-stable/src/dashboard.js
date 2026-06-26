// ── Admin dashboard chrome: timer, clock, AI report panel, incident strip ──
// Pulls incidents from the Go server (GET /incidents). Falls back to a demo
// incident so the dashboard always looks populated before the backend is wired.

const API_BASE   = 'http://localhost:8080'
const POLL_MS    = 2000

const timerEl    = document.getElementById('session-timer')
const clockEl    = document.getElementById('clock')
const reportBody = document.getElementById('report-body')
const stripEl    = document.getElementById('incident-strip')
const navCountEl = document.getElementById('nav-incident-count')
const stripNext  = document.getElementById('strip-next')

const reportExpand  = document.getElementById('report-expand')
const reportPdf     = document.getElementById('report-pdf')
const modal         = document.getElementById('report-modal')
const modalBody     = document.getElementById('modal-body')
const modalTitle    = document.getElementById('modal-title')
const modalClose    = document.getElementById('modal-close')
const modalBackdrop = modal ? modal.querySelector('.modal-backdrop') : null

let _selectedId = null
let _incidents  = []

// ── Session timer (counts up from page load) ───────────────────────────────
const _t0 = Date.now()
function tickTimer() {
  const s = Math.floor((Date.now() - _t0) / 1000)
  const hh = String(Math.floor(s / 3600)).padStart(2, '0')
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0')
  const ss = String(s % 60).padStart(2, '0')
  timerEl.textContent = `${hh}:${mm}:${ss}`
}
setInterval(tickTimer, 1000); tickTimer()

// ── Wall clock ─────────────────────────────────────────────────────────────
function tickClock() {
  clockEl.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}
setInterval(tickClock, 15000); tickClock()

// ── Helpers ────────────────────────────────────────────────────────────────
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c]))
const fmtTime = ts => {
  const d = ts ? new Date(ts) : new Date()
  return isNaN(d) ? '--:--' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}
// Human-readable trigger reason
const triggerLabel = t => t === 'intruder_detected'
  ? 'Second person detected in frame'
  : 'Hand visible + gaze toward wrist'

// 3-state verdict: confirmed cheating / needs human review / cleared
const verdictState = inc => Number(inc.cheating) === 1
  ? 'cheat'
  : (inc.review_required === true ? 'review' : 'clear')
const verdictText = st => st === 'cheat'
  ? 'CHEATING DETECTED'
  : st === 'review' ? 'NEEDS REVIEW' : 'NO CHEATING'

// Per-frame evidence table — exactly what the AI saw in each of the 6 frames
function frameTable(frames) {
  if (!Array.isArray(frames) || !frames.length) return ''
  const rows = frames.map(f => {
    const obj  = (f.object_in_hand || 'none')
    const warn = (obj && obj !== 'none') || f.second_person
    const sp   = f.second_person ? 'YES' : '—'
    return `<tr class="${warn ? 'warn' : ''}"><td>${esc(f.frame ?? '')}</td><td>${esc(obj)}</td><td>${esc(f.gaze || '--')}</td><td>${sp}</td><td>${esc(f.description || '')}</td></tr>`
  }).join('')
  return `<div class="gaze-table-wrap">
    <div class="rep-k" style="margin-bottom:7px">Per-Frame Evidence (what the AI saw)</div>
    <table class="gaze-table">
      <thead><tr><th>#</th><th>Object in hand</th><th>Gaze</th><th>2nd person</th><th>Notes</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`
}

// ── Render the AI report panel from a report object ────────────────────────
function renderReport(r) {
  if (reportExpand) reportExpand.disabled = !r
  if (reportPdf)    reportPdf.disabled    = !r
  if (!r) {
    reportBody.innerHTML = `
      <div class="report-empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="10"/></svg>
        <p>No incident selected.</p>
        <span>The AI report appears here when an incident fires.</span>
      </div>`
    return
  }

  const st    = verdictState(r)
  const cheat = st === 'cheat'
  const conf  = r.confidence_score != null ? `${r.confidence_score}%` : '--'
  const like  = (r.cheating_likelihood || 'low').toLowerCase()
  const mats  = Array.isArray(r.materials) ? r.materials : []
  const behs  = Array.isArray(r.suspicious_behaviors) ? r.suspicious_behaviors : []

  const verdictIcon = (st === 'cheat' || st === 'review')
    ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>`
    : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M20 6 9 17l-5-5"/></svg>`

  const matsHtml = mats.length
    ? mats.map(m => `<span class="tag">${esc(m)}</span>`).join('')
    : `<span class="tag none">None detected</span>`

  const behsHtml = behs.length
    ? `<ul class="beh-list">${behs.map(b => `<li>${esc(b)}</li>`).join('')}</ul>`
    : `<div class="rep-para">No suspicious behaviors flagged.</div>`

  reportBody.innerHTML = `
    <div class="verdict ${st}">
      ${verdictIcon}
      <span>${verdictText(st)}</span>
      <span class="conf">${conf}</span>
    </div>
    ${r.summary ? `<div class="rep-para" style="margin:-6px 0 12px;font-weight:600">${esc(r.summary)}</div>` : ''}

    <div class="rep-row">
      <span class="rep-k">Likelihood</span>
      <span class="rep-likelihood ${like}">${esc(like.toUpperCase())}</span>
    </div>

    <div class="rep-row">
      <span class="rep-k">Materials Detected</span>
      <div class="tags">${matsHtml}</div>
    </div>

    <div class="rep-row">
      <span class="rep-k">Suspicious Behaviors</span>
      ${behsHtml}
    </div>

    ${r.intruder_detected ? `
    <div class="rep-row">
      <span class="rep-k">Intruder / People</span>
      <div class="rep-para" style="color:#dc2626;font-weight:700">⚠ SECOND PERSON DETECTED — ${esc(r.face_count || 2)} people in frame</div>
    </div>` : `
    <div class="rep-row">
      <span class="rep-k">People in Frame</span>
      <div class="rep-para">${esc(r.face_count || 1)} (candidate only)</div>
    </div>`}

    ${r.phone_visible ? `
    <div class="rep-row">
      <span class="rep-k">Phone Detection</span>
      <div class="rep-para" style="color:#dc2626;font-weight:700">PHONE VISIBLE — confirmed by AI vision</div>
    </div>` : ''}

    <div class="rep-row">
      <span class="rep-k">Trigger</span>
      <div class="rep-para">${esc(triggerLabel(r.trigger_reason))}</div>
    </div>

    ${r.gaze_analysis ? `
    <div class="rep-row">
      <span class="rep-k">Gaze Analysis</span>
      <div class="rep-para">${esc(r.gaze_analysis)}</div>
    </div>` : ''}

    ${Array.isArray(r.per_frame) && r.per_frame.length ? `
    <div class="rep-row">${frameTable(r.per_frame)}</div>` : (r.visual_evidence ? `
    <div class="rep-row">
      <span class="rep-k">Visual Evidence</span>
      <div class="rep-para">${esc(r.visual_evidence)}</div>
    </div>` : '')}

    ${r.recommendation ? `
    <div class="rep-row">
      <div class="rep-reco"><b>Recommendation:</b> ${esc(r.recommendation)}</div>
    </div>` : ''}
  `
}

// ── Render the incident strip ──────────────────────────────────────────────
function isPending(inc) {
  const like = (inc.cheating_likelihood || '').toLowerCase()
  return like === 'analyzing' || like === 'unknown' || like === ''
}
// Surfaced on the strip: confirmed cheating, an intruder, a confirmed phone, OR
// a "needs review" ambiguous case (so a borderline frame is never silently dropped).
function isFlagged(inc) {
  return Number(inc.cheating) === 1 || inc.intruder_detected === true
    || inc.phone_visible === true || inc.review_required === true
}

function renderIncidents(list) {
  _incidents = list || []

  // Show confirmed violations (cheating / intruder / phone) or still-pending — hide cleared
  const displayed = _incidents.filter(i => isFlagged(i) || isPending(i))

  const flaggedCount = _incidents.filter(isFlagged).length
  navCountEl.textContent = flaggedCount
  navCountEl.dataset.zero = flaggedCount === 0 ? '1' : '0'

  if (!displayed.length) {
    stripEl.innerHTML = '<div class="strip-empty">No flagged incidents — monitoring…</div>'
    // If selected was cleared, deselect
    if (_selectedId && !displayed.find(i => i.id === _selectedId)) {
      _selectedId = null
      renderReport(null)
    }
    return
  }

  // If selected incident was cleared (cheating=0 analysis came back), pick first displayed
  if (_selectedId && !displayed.find(i => String(i.id) === String(_selectedId))) {
    _selectedId = displayed[0].id
  }
  if (_selectedId == null) _selectedId = displayed[0].id

  stripEl.innerHTML = displayed.map(inc => {
    const cheat    = Number(inc.cheating) === 1
    const intruder = inc.intruder_detected === true
    const review   = inc.review_required === true
    const pending  = isPending(inc)
    const thumb = inc.collage
      ? `<img class="inc-thumb" src="${esc(inc.collage)}" alt="incident">`
      : `<div class="inc-thumb-grid"><div>gaze</div><div>gaze</div><div>gaze</div><div>gaze</div></div>`
    const dotCls   = pending ? '' : (cheat || intruder) ? 'cheat' : review ? 'review' : 'clear'
    const labelTxt = pending ? '…' : intruder ? 'INTRUDER' : cheat ? 'CHEAT' : inc.phone_visible ? 'PHONE' : review ? 'REVIEW' : 'CLEAR'
    return `
      <div class="inc-card ${inc.id === _selectedId ? 'selected' : ''}" data-id="${esc(inc.id)}">
        ${thumb}
        <div class="inc-foot">
          <span class="inc-dot ${dotCls}"></span>
          <span class="inc-label">${labelTxt}</span>
          <span class="inc-time">${fmtTime(inc.timestamp)}</span>
        </div>
      </div>`
  }).join('')

  stripEl.querySelectorAll('.inc-card').forEach(el => {
    el.addEventListener('click', () => selectIncident(el.dataset.id))
  })

  const sel = displayed.find(i => String(i.id) === String(_selectedId)) || displayed[0]
  renderReport(sel)
}

function selectIncident(id) {
  _selectedId = id
  stripEl.querySelectorAll('.inc-card').forEach(el =>
    el.classList.toggle('selected', String(el.dataset.id) === String(id)))
  const inc = _incidents.find(i => String(i.id) === String(id))
  if (inc) renderReport(inc)
}

stripNext?.addEventListener('click', () => stripEl.scrollBy({ left: 340, behavior: 'smooth' }))

// ── Full report modal (expand button) ──────────────────────────────────────
function gazeTable(frames) {
  if (!Array.isArray(frames) || !frames.length) return ''
  const num = v => (typeof v === 'number' ? (v >= 0 ? '+' : '') + v.toFixed(2) : '--')
  const rows = frames.map(f => {
    const t = f.time
      ? new Date(f.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : '--'
    const gaze = f.gaze || '--'
    const warn = gaze !== 'AHEAD' && gaze !== 'UNKNOWN'
    return `<tr class="${warn ? 'warn' : ''}"><td>${f.frame ?? ''}</td><td>${t}</td><td>${esc(gaze)}</td><td>${num(f.iris_h)}</td><td>${num(f.iris_v)}</td><td>${num(f.head_yaw)}</td><td>${num(f.head_pitch)}</td></tr>`
  }).join('')
  return `<div class="gaze-table-wrap">
    <div class="rep-k" style="margin-bottom:7px">Per-Frame Gaze Data</div>
    <table class="gaze-table">
      <thead><tr><th>#</th><th>Time</th><th>Gaze</th><th>Iris H</th><th>Iris V</th><th>Yaw</th><th>Pitch</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`
}

function renderModal(inc) {
  const st    = verdictState(inc)
  const cheat = st === 'cheat'
  const conf  = inc.confidence_score != null ? `${inc.confidence_score}%` : '--'
  const like  = (inc.cheating_likelihood || 'low').toLowerCase()
  const mats  = Array.isArray(inc.materials) ? inc.materials : []
  const behs  = Array.isArray(inc.suspicious_behaviors) ? inc.suspicious_behaviors : []

  modalTitle.textContent = `Incident · ${fmtTime(inc.timestamp)}`
  modalBody.innerHTML = `
    <div class="modal-grid">
      <div class="modal-left">
        ${inc.collage
          ? `<img class="modal-collage" src="${esc(inc.collage)}" alt="evidence collage">`
          : `<div class="modal-collage-empty">No collage image</div>`}
        ${Array.isArray(inc.per_frame) && inc.per_frame.length ? frameTable(inc.per_frame) : ''}
        ${gazeTable(inc.gaze_data)}
      </div>
      <div class="modal-right">
        <div class="verdict ${st}">
          <span>${verdictText(st)}</span>
          <span class="conf">${conf}</span>
        </div>
        ${inc.summary ? `<div class="rep-para modal-summary">${esc(inc.summary)}</div>` : ''}
        <div class="rep-row"><span class="rep-k">Likelihood</span>
          <span class="rep-likelihood ${like}">${esc(like.toUpperCase())}</span></div>
        <div class="rep-row"><span class="rep-k">Materials Detected</span>
          <div class="tags">${mats.length ? mats.map(m => `<span class="tag">${esc(m)}</span>`).join('') : '<span class="tag none">None detected</span>'}</div></div>
        <div class="rep-row"><span class="rep-k">Suspicious Behaviors</span>
          ${behs.length ? `<ul class="beh-list">${behs.map(b => `<li>${esc(b)}</li>`).join('')}</ul>` : '<div class="rep-para">None flagged.</div>'}</div>
        ${inc.intruder_detected
          ? `<div class="rep-row"><span class="rep-k">Intruder / People</span><div class="rep-para" style="color:#dc2626;font-weight:700">⚠ SECOND PERSON DETECTED — ${esc(inc.face_count || 2)} people in frame</div></div>`
          : `<div class="rep-row"><span class="rep-k">People in Frame</span><div class="rep-para">${esc(inc.face_count || 1)} (candidate only)</div></div>`}
        ${inc.phone_visible ? `<div class="rep-row"><span class="rep-k">Phone Detection</span><div class="rep-para" style="color:#dc2626;font-weight:700">PHONE VISIBLE — confirmed by AI vision</div></div>` : ''}
        <div class="rep-row"><span class="rep-k">Trigger</span><div class="rep-para">${esc(triggerLabel(inc.trigger_reason))}</div></div>
        ${inc.gaze_analysis ? `<div class="rep-row"><span class="rep-k">Gaze Analysis</span><div class="rep-para">${esc(inc.gaze_analysis)}</div></div>` : ''}
        ${inc.visual_evidence ? `<div class="rep-row"><span class="rep-k">Visual Evidence</span><div class="rep-para">${esc(inc.visual_evidence)}</div></div>` : ''}
        ${inc.recommendation ? `<div class="rep-row"><div class="rep-reco"><b>Recommendation:</b> ${esc(inc.recommendation)}</div></div>` : ''}
      </div>
    </div>`
}

function openModal() {
  const inc = _incidents.find(i => String(i.id) === String(_selectedId))
  if (!inc) return
  renderModal(inc)
  modal.classList.remove('hidden')
}
function closeModal() { modal && modal.classList.add('hidden') }

reportExpand?.addEventListener('click', openModal)
modalClose?.addEventListener('click', closeModal)
modalBackdrop?.addEventListener('click', closeModal)
window.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal() })

// ── PDF helpers (real downloaded PDF, no print dialog) ──────────────────────
function waitForImages(doc) {
  return new Promise(res => {
    const imgs = Array.from(doc.images || [])
    if (!imgs.length) return res()
    let n = 0
    const done = () => { if (++n >= imgs.length) res() }
    imgs.forEach(im => im.complete ? done() : (im.onload = im.onerror = done))
    setTimeout(res, 4000)
  })
}

// Render the report living inside `iframe` to a downloaded PDF file.
async function renderIframeToPdf(iframe, filename) {
  if (typeof html2pdf === 'undefined') {
    alert('PDF engine is still loading — please wait a second and try again.')
    return false
  }
  const doc = iframe.contentDocument || iframe.contentWindow.document
  await waitForImages(doc)
  try {
    await html2pdf().set({
      margin:      0,
      filename:    filename,
      image:       { type: 'jpeg', quality: 0.95 },
      html2canvas: { scale: 2, useCORS: true, backgroundColor: '#ffffff', logging: false },
      jsPDF:       { unit: 'mm', format: 'a4', orientation: 'portrait' },
      pagebreak:   { mode: ['css', 'legacy'] }
    }).from(doc.body).save()
    return true
  } catch (e) {
    console.error('[pdf] generation failed:', e)
    alert('PDF generation failed: ' + e.message)
    return false
  }
}

// Direct download via a hidden off-screen iframe (used for per-incident PDF).
async function downloadReportPdf(fullHtml, filename) {
  const iframe = document.createElement('iframe')
  iframe.style.cssText = 'position:fixed;left:-99999px;top:0;width:820px;height:1160px;border:0'
  document.body.appendChild(iframe)
  try {
    const doc = iframe.contentDocument || iframe.contentWindow.document
    doc.open(); doc.write(fullHtml); doc.close()
    return await renderIframeToPdf(iframe, filename)
  } finally {
    setTimeout(() => iframe.remove(), 500)
  }
}

// Show the report ON SCREEN in a preview overlay first, with a Download button.
function previewReport(fullHtml, filename) {
  document.getElementById('report-preview-overlay')?.remove()

  const overlay = document.createElement('div')
  overlay.id = 'report-preview-overlay'
  overlay.style.cssText = 'position:fixed;inset:0;z-index:9999;background:rgba(15,23,42,.82);display:flex;flex-direction:column'

  const bar = document.createElement('div')
  bar.style.cssText = 'display:flex;align-items:center;gap:14px;padding:12px 20px;background:#0f172a;color:#fff;flex-shrink:0;box-shadow:0 2px 12px rgba(0,0,0,.45)'
  bar.innerHTML = `
    <span style="font-weight:800;color:#f59e0b;font-size:14px;letter-spacing:.5px">SESSION REPORT — PREVIEW</span>
    <span style="font-size:12px;color:#94a3b8">${esc(filename)}</span>
    <div style="margin-left:auto;display:flex;gap:10px">
      <button id="rp-dl" style="background:#f59e0b;color:#0f172a;border:0;border-radius:7px;padding:8px 18px;font-weight:800;cursor:pointer;font-size:13px">⬇ Download PDF</button>
      <button id="rp-close" style="background:#334155;color:#fff;border:0;border-radius:7px;padding:8px 14px;font-weight:600;cursor:pointer;font-size:13px">✕ Close</button>
    </div>`

  const scroll = document.createElement('div')
  scroll.style.cssText = 'flex:1;overflow:auto;display:flex;justify-content:center;padding:22px'

  const iframe = document.createElement('iframe')
  iframe.style.cssText = 'width:820px;max-width:100%;min-height:600px;border:0;background:#fff;box-shadow:0 10px 40px rgba(0,0,0,.55);border-radius:4px'
  scroll.appendChild(iframe)

  overlay.appendChild(bar)
  overlay.appendChild(scroll)
  document.body.appendChild(overlay)

  const doc = iframe.contentDocument || iframe.contentWindow.document
  doc.open(); doc.write(fullHtml); doc.close()
  const fit = () => { try { iframe.style.height = Math.max(600, doc.body.scrollHeight + 24) + 'px' } catch (_) {} }
  iframe.onload = fit; setTimeout(fit, 300); setTimeout(fit, 1200)

  const close = () => { overlay.remove(); document.removeEventListener('keydown', onKey) }
  const onKey = e => { if (e.key === 'Escape') close() }
  document.getElementById('rp-close').onclick = close
  overlay.addEventListener('click', e => { if (e.target === overlay) close() })
  document.addEventListener('keydown', onKey)

  document.getElementById('rp-dl').onclick = async () => {
    const b = document.getElementById('rp-dl')
    const t = b.textContent
    b.disabled = true; b.textContent = 'Generating…'
    await renderIframeToPdf(iframe, filename)
    b.disabled = false; b.textContent = t
  }
}

// ── PDF export ─────────────────────────────────────────────────────────────
async function downloadPDF() {
  const inc = _incidents.find(i => String(i.id) === String(_selectedId))
  if (!inc) return

  // Fetch collage as base64 so it embeds cleanly in the print window
  let collageB64 = ''
  if (inc.collage) {
    try {
      const r = await fetch(inc.collage)
      const blob = await r.blob()
      collageB64 = await new Promise(res => {
        const fr = new FileReader(); fr.onload = e => res(e.target.result); fr.readAsDataURL(blob)
      })
    } catch (_) {}
  }

  const st    = verdictState(inc)
  const cheat = st === 'cheat'
  const conf  = inc.confidence_score != null ? `${inc.confidence_score}%` : '--'
  const like  = (inc.cheating_likelihood || 'low').toLowerCase()
  const mats  = Array.isArray(inc.materials) ? inc.materials : []
  const behs  = Array.isArray(inc.suspicious_behaviors) ? inc.suspicious_behaviors : []
  const ts    = inc.timestamp ? new Date(inc.timestamp).toLocaleString() : '--'
  const now   = new Date().toLocaleString()

  const e = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))

  // Per-frame evidence rows (what the AI saw in each frame)
  const frameRows = Array.isArray(inc.per_frame) ? inc.per_frame.map(f => {
    const obj  = f.object_in_hand || 'none'
    const warn = (obj && obj !== 'none') || f.second_person
    return `<tr class="${warn ? 'warn' : ''}"><td>${e(f.frame ?? '')}</td><td>${e(obj)}</td><td>${e(f.gaze || '--')}</td><td>${f.second_person ? 'YES' : '—'}</td><td style="font-family:'Segoe UI',Arial,sans-serif">${e(f.description || '')}</td></tr>`
  }).join('') : ''

  const matsHtml = mats.length
    ? mats.map(m => `<span class="tag">${e(m)}</span>`).join(' ')
    : '<span style="color:#94a3b8">None detected</span>'

  const behsHtml = behs.length
    ? `<ul class="behaviors">${behs.map(b => `<li>${e(b)}</li>`).join('')}</ul>`
    : '<span style="color:#94a3b8">No suspicious behaviors flagged.</span>'

  const gazeRows = Array.isArray(inc.gaze_data) ? inc.gaze_data.map(f => {
    const t = f.time ? new Date(f.time).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}) : '--'
    const g = f.gaze || f.dir || '--'
    const warn = g !== 'AHEAD' && g !== 'UNKNOWN'
    const n = v => (typeof v === 'number' ? (v >= 0 ? '+' : '') + v.toFixed(2) : '--')
    return `<tr class="${warn ? 'warn' : ''}">
      <td>${f.frame ?? ''}</td><td>${t}</td><td>${e(g)}</td>
      <td>${n(f.iris_h ?? f.h)}</td><td>${n(f.iris_v ?? f.v)}</td>
      <td>${n(f.head_yaw ?? f.yaw)}</td><td>${n(f.head_pitch ?? f.pitch)}</td>
    </tr>`
  }).join('') : ''

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Proctor AI — Incident Report ${e(inc.id)}</title>
<style>
@page { size: A4 portrait; margin: 16mm 14mm 14mm; }
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Segoe UI',Arial,sans-serif;font-size:11pt;color:#1e293b;background:#fff;}
.hd{background:#1e293b;color:#fff;padding:14px 20px;display:flex;justify-content:space-between;align-items:flex-start;}
.hd-logo{font-size:15pt;font-weight:800;color:#f59e0b;letter-spacing:-.5px;}
.hd-logo small{display:block;font-size:9pt;font-weight:400;color:#94a3b8;margin-top:2px;}
.hd-meta{text-align:right;font-size:8.5pt;color:#94a3b8;line-height:1.6;}
.verdict{padding:11px 20px;display:flex;align-items:center;gap:12px;border-left:6px solid;}
.verdict.cheat{background:#fef2f2;border-color:#dc2626;}
.verdict.review{background:#fffbeb;border-color:#d97706;}
.verdict.clear{background:#f0fdf4;border-color:#16a34a;}
.v-icon{width:22px;height:22px;flex-shrink:0;}
.v-label{font-size:14pt;font-weight:800;}
.verdict.cheat .v-label{color:#dc2626;}
.verdict.review .v-label{color:#d97706;}
.verdict.clear .v-label{color:#16a34a;}
.v-conf{font-size:10pt;color:#64748b;margin-left:auto;}
.summary{padding:9px 20px;background:#f8fafc;border-bottom:1px solid #e2e8f0;font-size:10pt;color:#475569;font-style:italic;line-height:1.5;}
.grid{display:grid;grid-template-columns:55% 45%;gap:0;}
.gl{padding:14px 14px 14px 20px;}
.gr{padding:14px 20px 14px 14px;border-left:1px solid #e2e8f0;}
.collage{width:100%;height:auto;border:1px solid #e2e8f0;border-radius:6px;display:block;}
.no-collage{height:160px;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:10pt;}
.fld{margin-bottom:11px;}
.fld-k{font-size:7.5pt;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px;}
.fld-v{font-size:10pt;color:#1e293b;line-height:1.5;}
.tag{display:inline-block;background:#fef3c7;color:#92400e;border:1px solid #fde68a;border-radius:10px;padding:1px 9px;font-size:9pt;margin:2px 3px 2px 0;}
.lk-high{color:#dc2626;font-weight:700;}
.lk-medium{color:#d97706;font-weight:700;}
.lk-low{color:#16a34a;font-weight:700;}
ul.behaviors{padding-left:15px;}
ul.behaviors li{font-size:10pt;margin-bottom:2px;}
.tbl-wrap{padding:0 20px 14px;}
.tbl-title{font-size:8pt;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;}
table{width:100%;border-collapse:collapse;font-size:8.5pt;font-family:'Consolas','Courier New',monospace;}
th{background:#f1f5f9;text-align:left;padding:5px 7px;border:1px solid #e2e8f0;color:#64748b;font-weight:600;}
td{padding:4px 7px;border:1px solid #e2e8f0;}
tr.warn td{background:#fef2f2;color:#991b1b;}
tr:nth-child(even):not(.warn) td{background:#f8fafc;}
.ft{padding:8px 20px;border-top:1px solid #e2e8f0;display:flex;justify-content:space-between;font-size:8pt;color:#94a3b8;}
@media print{body{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
</style>
</head>
<body>

<div class="hd">
  <div class="hd-logo">PROCTOR AI
    <small>AI Exam Proctoring System — Incident Report</small>
  </div>
  <div class="hd-meta">
    <div>Incident ID: <b style="color:#e2e8f0">${e(inc.id)}</b></div>
    <div>Captured: ${e(ts)}</div>
    <div>Report generated: ${e(now)}</div>
  </div>
</div>

<div class="verdict ${st}">
  ${st === 'clear'
    ? `<svg class="v-icon" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg>`
    : `<svg class="v-icon" viewBox="0 0 24 24" fill="none" stroke="${st === 'cheat' ? '#dc2626' : '#d97706'}" stroke-width="2.5"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/></svg>`
  }
  <span class="v-label">${st === 'cheat' ? 'CHEATING DETECTED' : st === 'review' ? 'NEEDS REVIEW' : 'NO CHEATING DETECTED'}</span>
  <span class="v-conf">Confidence: <b>${e(conf)}</b></span>
</div>

${inc.summary ? `<div class="summary">${e(inc.summary)}</div>` : ''}

<div class="grid">
  <div class="gl">
    ${collageB64
      ? `<img class="collage" src="${collageB64}" alt="Evidence collage">`
      : `<div class="no-collage">No collage available</div>`}
  </div>
  <div class="gr">
    <div class="fld">
      <div class="fld-k">Verdict</div>
      <div class="fld-v"><span class="lk-${like}">${e(like.toUpperCase())}</span> likelihood &nbsp;·&nbsp; ${e(conf)} confidence</div>
    </div>
    <div class="fld">
      <div class="fld-k">Materials Detected</div>
      <div class="fld-v">${matsHtml}</div>
    </div>
    <div class="fld">
      <div class="fld-k">Suspicious Behaviors</div>
      <div class="fld-v">${behsHtml}</div>
    </div>
    ${inc.gaze_analysis ? `<div class="fld"><div class="fld-k">Gaze Analysis</div><div class="fld-v">${e(inc.gaze_analysis)}</div></div>` : ''}
    ${inc.visual_evidence ? `<div class="fld"><div class="fld-k">Visual Evidence</div><div class="fld-v">${e(inc.visual_evidence)}</div></div>` : ''}
    ${inc.intruder_detected
      ? `<div class="fld"><div class="fld-k" style="color:#dc2626">Intruder / People in Frame</div><div class="fld-v" style="color:#dc2626;font-weight:700">⚠ SECOND PERSON DETECTED — ${e(inc.face_count || 2)} people present</div></div>`
      : `<div class="fld"><div class="fld-k">People in Frame</div><div class="fld-v">${e(inc.face_count || 1)} (candidate only)</div></div>`}
    ${inc.phone_visible ? `<div class="fld"><div class="fld-k" style="color:#dc2626">Phone Detection</div><div class="fld-v" style="color:#dc2626;font-weight:700">PHONE VISIBLE — AI confirmed</div></div>` : ''}
    <div class="fld"><div class="fld-k">Trigger</div><div class="fld-v">${e(triggerLabel(inc.trigger_reason))}</div></div>
    ${inc.recommendation ? `<div class="fld"><div class="fld-k">Recommendation</div><div class="fld-v" style="color:#92400e;font-weight:600">${e(inc.recommendation)}</div></div>` : ''}
  </div>
</div>

${frameRows ? `
<div class="tbl-wrap">
  <div class="tbl-title">Per-Frame Evidence — What the AI Saw</div>
  <table>
    <thead><tr><th>#</th><th>Object in Hand</th><th>Gaze</th><th>2nd Person</th><th>Notes</th></tr></thead>
    <tbody>${frameRows}</tbody>
  </table>
</div>` : ''}

${gazeRows ? `
<div class="tbl-wrap">
  <div class="tbl-title">Per-Frame Gaze Data</div>
  <table>
    <thead><tr><th>#</th><th>Timestamp</th><th>Gaze</th><th>Iris H</th><th>Iris V</th><th>Head Yaw</th><th>Head Pitch</th></tr></thead>
    <tbody>${gazeRows}</tbody>
  </table>
</div>` : ''}

<div class="ft">
  <span>Generated by <b>Proctor AI</b> · Groq Llama 4 Scout vision</span>
  <span>Trigger: ${e(triggerLabel(inc.trigger_reason))}</span>
</div>

</body>
</html>`

  await downloadReportPdf(html, `Proctor-Incident-${inc.id}.pdf`)
}

document.getElementById('report-pdf')?.addEventListener('click', downloadPDF)
document.getElementById('modal-pdf')?.addEventListener('click', downloadPDF)

// ── Session Report (End Session button) ────────────────────────────────────
async function generateSessionPDF() {
  const btn = document.getElementById('end-session-btn')
  const resetBtn = () => { if (btn) { btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9l6 6M15 9l-6 6"/></svg> End Session'; btn.disabled = false } }
  if (btn) { btn.textContent = 'Finishing capture…'; btn.disabled = true }

  // 1) Wait for any in-flight capture/upload so the LAST incident is included
  //    (fixes the race where End Session ran before the final 6 frames landed).
  const waitStart = Date.now()
  while (window.__captureBusy && Date.now() - waitStart < 12000) {
    await new Promise(r => setTimeout(r, 200))
  }

  // 2) Fetch the session — give pending LLM verdicts a bounded chance to resolve
  //    so the report shows real verdicts, not "pending". Never hangs (capped ~7.5s).
  if (btn) btn.textContent = 'Analyzing…'
  let session
  try {
    for (let attempt = 0; attempt < 6; attempt++) {
      const r = await fetch(`${API_BASE}/session-report`, { cache: 'no-store' })
      session = await r.json()
      if (!session || !(Number(session.pending_count) > 0) || attempt === 5) break
      await new Promise(res => setTimeout(res, 1500))
    }
  } catch (_) {
    alert('Could not reach the API server.')
    resetBtn()
    return
  }
  if (btn) btn.textContent = 'Generating…'

  // Include confirmed violations (cheating / intruder / phone) + still-pending incidents
  const allIncidents = Array.isArray(session.incidents) ? session.incidents : []
  const incidents = allIncidents.filter(i => {
    const like = (i.cheating_likelihood || '').toLowerCase()
    const pending = like === 'analyzing' || like === 'unknown' || like === ''
    const flagged = Number(i.cheating) === 1 || i.intruder_detected === true || i.phone_visible === true
    return flagged || pending
  })

  if (incidents.length === 0) {
    alert('No violations found — candidate was clean this session (no cheating, no intruders, no phone).')
    resetBtn()
    return
  }

  // Fetch all collages as base64 in parallel
  const collageMap = {}
  await Promise.all(incidents.map(async inc => {
    if (!inc.collage) return
    try {
      const r = await fetch(inc.collage)
      const blob = await r.blob()
      collageMap[inc.id] = await new Promise(res => {
        const fr = new FileReader(); fr.onload = e => res(e.target.result); fr.readAsDataURL(blob)
      })
    } catch (_) {}
  }))

  const now      = new Date().toLocaleString()
  // Real session ID derived from the session start (first incident, else now)
  const _sd      = session.first_incident ? new Date(session.first_incident) : new Date()
  const _pad     = n => String(n).padStart(2, '0')
  const sessionId = `PA-${_sd.getFullYear()}${_pad(_sd.getMonth()+1)}${_pad(_sd.getDate())}-${_pad(_sd.getHours())}${_pad(_sd.getMinutes())}`
  const verdict  = (session.overall_verdict || 'clean').toUpperCase()
  const vColor   = verdict === 'CHEATING' ? '#dc2626' : verdict === 'SUSPICIOUS' ? '#d97706' : verdict === 'PENDING' ? '#64748b' : '#16a34a'
  const vBg      = verdict === 'CHEATING' ? '#fef2f2' : verdict === 'SUSPICIOUS' ? '#fffbeb' : verdict === 'PENDING' ? '#f1f5f9' : '#f0fdf4'
  const e = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))
  const fmtTs = ts => ts ? new Date(ts).toLocaleString() : '--'
  const fmtTime = ts => ts ? new Date(ts).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}) : '--'
  const durSecs = Number(session.duration_secs || 0)
  const durStr  = durSecs > 0
    ? `${Math.floor(durSecs/60)}m ${durSecs%60}s`
    : '--'
  const intruderTotal = incidents.filter(i => i.intruder_detected === true).length

  const incRows = incidents.map((inc, idx) => {
    const like     = (inc.cheating_likelihood || 'analyzing').toLowerCase()
    const isPending = like === 'analyzing' || like === 'unknown' || like === ''
    const cheat    = !isPending && Number(inc.cheating) === 1
    const intruder = inc.intruder_detected === true
    const review   = !isPending && !cheat && inc.review_required === true
    const conf     = !isPending && inc.confidence_score != null ? `${inc.confidence_score}%` : '--'
    const mats     = Array.isArray(inc.materials) && inc.materials.length ? inc.materials.join(', ') : 'None'
    const behs     = Array.isArray(inc.suspicious_behaviors) && inc.suspicious_behaviors.length
      ? `<ul style="padding-left:14px;margin:4px 0">${inc.suspicious_behaviors.map(b => `<li>${e(b)}</li>`).join('')}</ul>`
      : 'None flagged'
    const img      = collageMap[inc.id]
      ? `<img src="${collageMap[inc.id]}" style="width:100%;border-radius:4px;border:1px solid #e2e8f0;margin-top:8px">`
      : ''
    // Per-frame evidence mini-table
    const frameTbl = Array.isArray(inc.per_frame) && inc.per_frame.length
      ? `<div style="margin-top:8px"><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block;margin-bottom:4px">Per-Frame Evidence</b>
         <table style="width:100%;border-collapse:collapse;font-size:8.5pt">
           <thead><tr><th style="background:#f1f5f9;border:1px solid #e2e8f0;padding:3px 6px;text-align:left">#</th><th style="background:#f1f5f9;border:1px solid #e2e8f0;padding:3px 6px;text-align:left">Object</th><th style="background:#f1f5f9;border:1px solid #e2e8f0;padding:3px 6px;text-align:left">Gaze</th><th style="background:#f1f5f9;border:1px solid #e2e8f0;padding:3px 6px;text-align:left">2nd</th><th style="background:#f1f5f9;border:1px solid #e2e8f0;padding:3px 6px;text-align:left">Notes</th></tr></thead>
           <tbody>${inc.per_frame.map(f => { const o=f.object_in_hand||'none'; const w=(o&&o!=='none')||f.second_person; return `<tr><td style="border:1px solid #e2e8f0;padding:3px 6px;${w?'background:#fef2f2;color:#991b1b':''}">${e(f.frame??'')}</td><td style="border:1px solid #e2e8f0;padding:3px 6px;${w?'background:#fef2f2;color:#991b1b':''}">${e(o)}</td><td style="border:1px solid #e2e8f0;padding:3px 6px">${e(f.gaze||'--')}</td><td style="border:1px solid #e2e8f0;padding:3px 6px">${f.second_person?'YES':'—'}</td><td style="border:1px solid #e2e8f0;padding:3px 6px">${e(f.description||'')}</td></tr>` }).join('')}</tbody>
         </table></div>`
      : ''
    const likeColor   = like === 'high' ? '#dc2626' : like === 'medium' ? '#d97706' : like === 'low' ? '#16a34a' : '#94a3b8'
    const violation   = cheat || intruder
    const statusText  = isPending ? '⏳ PENDING ANALYSIS' : intruder ? '⚠ INTRUDER DETECTED' : cheat ? '⚠ CHEATING DETECTED' : review ? '⚑ NEEDS REVIEW' : '✓ NO CHEATING'
    const statusColor = isPending ? '#d97706' : violation ? '#dc2626' : review ? '#d97706' : '#16a34a'
    const rowBorder   = isPending ? '#e2e8f0' : violation ? '#fca5a5' : review ? '#fcd34d' : '#86efac'
    return `
    <div style="border:1.5px solid ${rowBorder};border-radius:8px;padding:14px 16px;margin-bottom:14px;page-break-inside:avoid">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap">
        <span style="background:#1e293b;color:#fff;border-radius:20px;padding:2px 10px;font-size:9pt;font-weight:700">#${idx+1}</span>
        <span style="font-size:10pt;color:#64748b">${fmtTs(inc.timestamp)}</span>
        <span style="font-size:9pt;color:#94a3b8">${e(triggerLabel(inc.trigger_reason))}</span>
        <span style="margin-left:auto;font-weight:800;color:${statusColor};font-size:10.5pt">${statusText}</span>
        ${!isPending ? `<span style="color:#64748b;font-size:10pt">${conf} confidence · <span style="color:${likeColor};font-weight:700">${like.toUpperCase()} likelihood</span></span>` : ''}
      </div>
      ${img}
      ${isPending ? `<div style="margin-top:10px;padding:8px 12px;background:#fef3c7;border-radius:6px;font-size:9.5pt;color:#92400e">⚠ AI analysis pending. Review collage manually.</div>` : ''}
      ${intruder ? `<div style="margin-top:8px;padding:6px 12px;background:#fef2f2;border-left:3px solid #dc2626;border-radius:0 4px 4px 0;font-size:9.5pt;color:#dc2626;font-weight:700">⚠ SECOND PERSON DETECTED — ${e(inc.face_count || 2)} people in frame</div>` : ''}
      ${inc.phone_visible ? `<div style="margin-top:8px;padding:6px 12px;background:#fef2f2;border-left:3px solid #dc2626;border-radius:0 4px 4px 0;font-size:9.5pt;color:#dc2626;font-weight:700">PHONE VISIBLE — AI confirmed</div>` : ''}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;font-size:9.5pt">
        <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase">Materials Detected</b><div style="margin-top:3px">${e(mats)}</div></div>
        <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase">Suspicious Behaviors</b><div style="margin-top:3px">${behs}</div></div>
      </div>
      ${inc.gaze_analysis ? `<div style="margin-top:8px;font-size:9.5pt;color:#475569"><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase">Gaze Analysis</b><div style="margin-top:3px">${e(inc.gaze_analysis)}</div></div>` : ''}
      ${frameTbl || (inc.visual_evidence ? `<div style="margin-top:8px;font-size:9.5pt;color:#475569"><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase">Visual Evidence</b><div style="margin-top:3px">${e(inc.visual_evidence)}</div></div>` : '')}
      ${inc.recommendation ? `<div style="margin-top:8px;font-size:9.5pt;background:#fffbeb;border-left:3px solid #f59e0b;padding:6px 10px;border-radius:0 4px 4px 0"><b>Recommendation:</b> ${e(inc.recommendation)}</div>` : ''}
    </div>`
  }).join('')

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Session Report — ${now}</title>
<style>
@page{size:A4 portrait;margin:16mm 14mm 14mm;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Segoe UI',Arial,sans-serif;font-size:11pt;color:#1e293b;background:#fff;}
@media print{body{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
</style>
</head>
<body>

<div style="background:#1e293b;color:#fff;padding:14px 20px;display:flex;justify-content:space-between;align-items:flex-start">
  <div>
    <div style="font-size:16pt;font-weight:800;color:#f59e0b;letter-spacing:-.5px">PROCTOR AI</div>
    <div style="font-size:9pt;color:#94a3b8;margin-top:2px">Candidate Session Report — Complete Exam Analysis</div>
  </div>
  <div style="text-align:right;font-size:8.5pt;color:#94a3b8;line-height:1.7">
    <div>Generated: <b style="color:#e2e8f0">${e(now)}</b></div>
    <div>Session ID: <b style="color:#e2e8f0">${e(sessionId)}</b></div>
    <div>First incident: ${e(fmtTs(session.first_incident))}</div>
    <div>Last incident: ${e(fmtTs(session.last_incident))}</div>
    <div>Span: <b style="color:#e2e8f0">${e(durStr)}</b></div>
  </div>
</div>

<div style="padding:12px 20px;background:${vBg};border-left:6px solid ${vColor};display:flex;align-items:center;gap:16px">
  <div style="font-size:16pt;font-weight:800;color:${vColor}">OVERALL VERDICT: ${e(verdict)}</div>
  <div style="font-size:10pt;color:#64748b;margin-left:auto;text-align:right;line-height:1.7">
    <div>${session.cheating_count || 0} of ${session.analyzed_count ?? session.total} analyzed incidents flagged</div>
    <div>Max confidence: <b>${session.max_confidence ?? '--'}%</b> · Highest: <b>${e((session.highest_likelihood||'low').toUpperCase())}</b></div>
    ${(session.pending_count||0) > 0 ? `<div style="color:#d97706">⚠ ${session.pending_count} incident(s) still pending AI analysis</div>` : ''}
  </div>
</div>

<div style="padding:10px 20px;background:#f8fafc;border-bottom:1px solid #e2e8f0;display:flex;gap:32px;font-size:10pt;flex-wrap:wrap">
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Flagged Incidents</b><span style="color:${incidents.length > 0 ? '#dc2626' : 'inherit'};font-weight:700">${incidents.length}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Confirmed Cheating</b><span style="color:${(session.cheating_count||0) > 0 ? '#dc2626' : '#16a34a'};font-weight:700">${session.cheating_count || 0}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Needs Review</b><span style="color:${(session.review_count||0) > 0 ? '#d97706' : 'inherit'};font-weight:700">${session.review_count || 0}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Phone Detections</b><span style="color:${incidents.filter(i=>i.phone_visible).length > 0 ? '#dc2626' : 'inherit'};font-weight:700">${incidents.filter(i=>i.phone_visible).length}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Intruder Incidents</b><span style="color:${intruderTotal > 0 ? '#dc2626' : 'inherit'};font-weight:700">${intruderTotal}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Max People in Frame</b><span style="color:${(session.max_people||1) > 1 ? '#dc2626' : 'inherit'};font-weight:700">${session.max_people ?? 1}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">AI Analyzed</b>${session.analyzed_count ?? session.total} / ${session.total}</div>
  ${(session.pending_count||0) > 0 ? `<div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Pending Review</b><span style="color:#d97706;font-weight:700">${session.pending_count} — check manually</span></div>` : ''}
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Total Frames</b>${session.total_frames ?? '--'}</div>
</div>

<div style="padding:14px 20px">
  <div style="font-size:8.5pt;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px">Incident Timeline (${incidents.length} recorded)</div>
  ${incRows}
</div>

<div style="padding:10px 20px;border-top:1px solid #e2e8f0;display:flex;justify-content:space-between;font-size:8pt;color:#94a3b8">
  <span>Generated by <b>Proctor AI</b> · Powered by Gemini 2.0 Flash / Groq</span>
  <span>For examiner use only — confidential</span>
</div>

</body></html>`

  previewReport(html, `Proctor-Session-${sessionId}.pdf`)
  resetBtn()
}

document.getElementById('end-session-btn')?.addEventListener('click', () => {
  if (confirm(`End this exam session and generate the complete candidate report?\n\nThis will compile all ${_incidents.length} recorded incident(s) into one PDF for the examiner.`)) {
    generateSessionPDF()
  }
})

// ── Poll the server for incidents (demo fallback while backend is wired) ──
async function poll() {
  try {
    const res = await fetch(`${API_BASE}/incidents`, { cache: 'no-store' })
    if (res.ok) {
      const data = await res.json()
      const list = Array.isArray(data) ? data : (data.incidents || [])
      renderIncidents(list)
      return
    }
  } catch (_) { /* server not reachable */ }
  renderIncidents([])
}
setInterval(poll, POLL_MS); poll()
