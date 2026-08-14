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
const triggerLabel = t =>
    t === 'intruder_detected'       ? 'Second person detected in frame'
  : t === 'phone_detected_realtime' ? 'Phone detected by real-time on-device detector'
  : t === 'covert_device_suspected' ? 'Repeated look-down — possible hidden device'
  : t === 'framing_torso_hidden'    ? 'Camera tilted up — torso & hands hidden from view'
  : 'Hand visible + gaze toward wrist'

// Phone-detection line reflecting BOTH signals: the real-time on-device detector
// (inc.phone_detected) and the AI vision model (inc.phone_visible). Returns
// { html } or '' when neither fired.
// ── Shared verdict/evidence wording ────────────────────────────────────────
// Used IDENTICALLY by the on-screen report, the modal, and BOTH PDF exports, so
// a downloaded report reads exactly like what was shown on the web.
function hasPhone(inc) {
  return inc.phone_ai_confirmed === true || inc.phone_detected === true || inc.phone_visible === true
}
function phoneText(inc) {
  const ai = inc.phone_ai_confirmed === true   // AI vision GENUINELY saw a phone
  const rt = inc.phone_detected === true        // on-device real-time detector
  return ai && rt ? 'PHONE CONFIRMED — flagged by the real-time detector AND confirmed by AI vision'
       : ai        ? 'PHONE CONFIRMED — confirmed by AI vision'
       :             'PHONE DETECTED — on-device real-time detector'
}
function intruderText(inc) {
  return `⚠ SECOND PERSON DETECTED — ${inc.face_count || 2} people in frame`
}
// EXACT mirror of the Go server classifyObject, so a per-frame row's highlighting
// always agrees with the server's verdict: whole-word tokens, multi-word device
// spellings normalised, clause-split with precedence cheat > benign > unknown >
// none (a device listed with a benign item still flags), negation = empty scene.
function classifyObjClient(o) {
  o = String(o || '').toLowerCase().trim()
  if (!o) return 'none'
  const bigrams = [['ear buds','earbud'],['ear bud','earbud'],['ear phones','earphone'],['ear phone','earphone'],['head phones','headphone'],['head phone','headphone'],['ear piece','earpiece'],['wrist watch','wristwatch'],['smart watch','smartwatch'],['cell phone','cellphone'],['air pods','airpod'],['air pod','airpod'],['ear pods','earpod'],['ear pod','earpod'],['smart glasses','smartglasses'],['smart phone','smartphone']]
  for (const [a,b] of bigrams) o = o.split(a).join(b)
  // "none" = everything that can NEVER be a held cheating aid: body parts,
  // clothing, the room/peripherals, and filler words. Skipped inside a clause,
  // so they can never clear a device mentioned next to them.
  const none = new Set(['none','empty','nothing','nil','na','unknown',
    'hand','hands','finger','fingers','fist','palm','face','chin','hair','shoulder','arm','wrist','elbow','neck','ear','head','torso','chest','body','back','lap','knee','leg','thumb','knuckle','nail','skin','forehead','eye','nose','mouth','lip','cheek','jaw','beard',
    'shirt','tshirt','sleeve','collar','hoodie','jacket','sweater','tie','scarf','clothes','clothing','fabric',
    'desk','table','chair','wall','ceiling','floor','window','door','background','room','bed','keyboard','mouse','mousepad','webcam','microphone','mic',
    'near','in','on','at','by','to','of','the','a','an','my','his','her','its','own','their','them','they','is','are','was','were','being','been','resting','held','holding','visible','front','up','down','left','right',
    'under','over','behind','above','below','against','around','between','beneath','inside','outside','into','onto','off','away','across','toward','towards','while','during',
    'wearing','worn','seen','shown','appears','appearing','looks','looking','raised','lowered','covering','covered','partially','partly','slightly',
    'nobody','noone','anyone','anything','else','alone','clear','all','only','just','also','still','from','that','this','there','what','out'])
  const benign = new Set(['bottle','water','cup','glass','mug','flask','tumbler','drink','coffee','tea','juice','soda','food','snack','fruit','tissue','napkin','cloth','towel','pen','pencil','marker','eraser','ruler','glasses','eyeglasses','spectacles','sunglasses'])
  const cheat = new Set(['phone','smartphone','iphone','cellphone','cell','mobile','android','tablet','ipad','laptop','computer','monitor','screen','device','earbud','earphone','headphone','headset','airpod','earpiece','watch','smartwatch','wristwatch','smartglasses','earpod','paper','note','notepad','sheet','page','card','flashcard','document','book','notebook','textbook','calculator','cheatsheet','chit'])
  const negation = new Set(['no','not','without'])
  const lookup = t => {
    if (none.has(t)) return 'none'
    if (benign.has(t)) return 'benign'
    if (cheat.has(t)) return 'cheat'
    if (t.length > 3 && t.endsWith('s')) {
      const s = t.slice(0, -1)
      if (none.has(s)) return 'none'
      if (benign.has(s)) return 'benign'
      if (cheat.has(s)) return 'cheat'
    }
    return ''
  }
  const classifyClause = c => {
    let cls = '', saw = false, neg = false
    for (const t of c.split(/[^a-z0-9]+/).filter(Boolean)) {
      if (negation.has(t)) { neg = true; continue }
      const r = lookup(t)
      if (r === 'benign') { cls = 'benign'; saw = true }
      else if (r === 'cheat') { cls = 'cheat'; saw = true }
      else if (r === '') { saw = true }
    }
    if (neg) return 'none'
    if (cls) return cls
    return saw ? 'unknown' : 'none'
  }
  let s = o
  for (const sep of [' and ',' but ',' or ',', ',',',' & ',' + ',' plus ',' with ',' next to ',' beside ',' near ','/']) s = s.split(sep).join('')
  const rank = { none: 0, benign: 1, unknown: 2, cheat: 3 }
  let best = 'none'
  for (let c of s.split('')) { c = c.trim(); if (!c) continue; const cl = classifyClause(c); if (rank[cl] > rank[best]) best = cl }
  return best
}
function isBenignObj(o) { return classifyObjClient(o) === 'benign' }
// A frame row is reddened when the server would treat the object as a cheat aid or
// an unidentified object (→ review), or a second person is present. Benign items
// and empty/negated scenes are never reddened — matching the server verdict exactly.
function objWarn(o, second) {
  const c = classifyObjClient(o)
  return c === 'cheat' || c === 'unknown' || second === true
}
// Canonical verdict label — INTRUDER is distinguished from generic cheating, and
// the exact same text is shown on the web panel and in every PDF.
function statusLabel(inc) {
  // A confirmed on-device violation (phone/intruder) is authoritative even while the
  // LLM enrichment is still pending — surface it, don't bury it behind "pending".
  if (inc.intruder_confirmed === true) return 'INTRUDER DETECTED'
  if (Number(inc.cheating) === 1) return 'CHEATING DETECTED'
  if (inc.llm_pending === true) return 'PENDING ANALYSIS'   // AI still working on it
  const like = (inc.cheating_likelihood || '').toLowerCase()
  if (like === 'analyzing' || like === 'unknown' || like === '') return 'PENDING ANALYSIS'
  if (inc.review_required === true) return 'NEEDS REVIEW'
  return 'NO CHEATING'
}

function phoneRowHtml(inc) {
  if (!hasPhone(inc)) return ''
  return `<div class="rep-row"><span class="rep-k">Phone Detection</span>
      <div class="rep-para" style="color:#dc2626;font-weight:700">${phoneText(inc)}</div></div>`
}

// Notes/paper detection — surfaced from the LLM materials (paper / book / notes / page).
function hasNotes(inc) {
  const mats = Array.isArray(inc.materials) ? inc.materials : []
  return mats.some(m => /paper|book|note|page|sheet|document/i.test(String(m)))
}
function notesRowHtml(inc) {
  if (!hasNotes(inc)) return ''
  return `<div class="rep-row"><span class="rep-k">Notes / Paper</span>
      <div class="rep-para" style="color:#dc2626;font-weight:700">📝 NOTES / PAPER DETECTED — confirmed by AI vision</div></div>`
}

// 3-state verdict: confirmed cheating / needs human review / cleared.
// An intruder is ALWAYS a red violation (never a green "clear"), matching the
// session PDF — so the verdict banner colour agrees across every surface.
const verdictState = inc => (Number(inc.cheating) === 1 || inc.intruder_confirmed === true)
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
    const warn = objWarn(obj, f.second_person)
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
      <span>${statusLabel(r)}</span>
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

    ${r.intruder_confirmed ? `
    <div class="rep-row">
      <span class="rep-k">Intruder / People</span>
      <div class="rep-para" style="color:#dc2626;font-weight:700">${intruderText(r)}</div>
    </div>` : `
    <div class="rep-row">
      <span class="rep-k">People in Frame</span>
      <div class="rep-para">Candidate only</div>
    </div>`}

    ${phoneRowHtml(r)}
    ${notesRowHtml(r)}

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
  if (inc.llm_pending === true) return true   // queued/being analyzed by AI right now
  const like = (inc.cheating_likelihood || '').toLowerCase()
  return like === 'analyzing' || like === 'unknown' || like === ''
}
// Surfaced on the strip: confirmed cheating, an intruder, a confirmed phone, OR
// a "needs review" ambiguous case (so a borderline frame is never silently dropped).
function isFlagged(inc) {
  return Number(inc.cheating) === 1 || inc.intruder_confirmed === true
    || hasPhone(inc) || inc.review_required === true
}

function renderIncidents(list) {
  _incidents = list || []

  // Show EVERY incident that fired (flagged, needs-review, and cleared) so the
  // live strip matches the downloadable report exactly — nothing is hidden.
  const displayed = _incidents.slice()

  const flaggedCount = _incidents.filter(isFlagged).length
  navCountEl.textContent = flaggedCount
  navCountEl.dataset.zero = flaggedCount === 0 ? '1' : '0'

  if (!displayed.length) {
    stripEl.innerHTML = '<div class="strip-empty">No incidents yet — monitoring…</div>'
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
    const intruder = inc.intruder_confirmed === true
    const review   = inc.review_required === true
    const pending  = isPending(inc)
    const thumb = inc.collage
      ? `<img class="inc-thumb" src="${esc(inc.collage)}" alt="incident">`
      : `<div class="inc-thumb-grid"><div>gaze</div><div>gaze</div><div>gaze</div><div>gaze</div></div>`
    const dotCls   = pending ? '' : (cheat || intruder) ? 'cheat' : review ? 'review' : 'clear'
    const labelTxt = pending ? '…' : intruder ? 'INTRUDER' : cheat ? 'CHEAT' : hasPhone(inc) ? 'PHONE' : review ? 'REVIEW' : 'CLEAR'
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

// ── Evidence-strength + examiner review (report layer) ─────────────────────
function sysVerdictLabel(v) {
  return ({
    confirmed_violation: 'Confirmed Violation',
    needs_review:        'Needs Review',
    no_violation:        'No Violation Detected',
    technical_failure:   'Technical Failure',
  })[v] || (v || '').replace(/_/g, ' ')
}

// evidenceRows renders the new report layer: system verdict, evidence strength,
// and the "why not clear / why not cheating" explanations that make the verdict
// defensible.
function evidenceRows(inc) {
  let h = ''
  if (inc.system_verdict)
    h += `<div class="rep-row"><span class="rep-k">System Verdict</span><div class="rep-para"><b>${esc(sysVerdictLabel(inc.system_verdict))}</b></div></div>`
  if (inc.evidence_strength) {
    const scoreTxt = inc.evidence_score != null ? ` · score ${esc(inc.evidence_score)}/100` : ''
    h += `<div class="rep-row"><span class="rep-k">Evidence Strength</span><span class="rep-likelihood ev-${esc(inc.evidence_strength)}">${esc(String(inc.evidence_strength).toUpperCase())}${scoreTxt}</span></div>`
  }
  if (inc.why_not_clear)
    h += `<div class="rep-row"><span class="rep-k">Why not clear?</span><div class="rep-para">${esc(inc.why_not_clear)}</div></div>`
  if (inc.why_not_cheating)
    h += `<div class="rep-row"><span class="rep-k">Why not cheating?</span><div class="rep-para">${esc(inc.why_not_cheating)}</div></div>`
  // FIX 5 — contradictions shown explicitly, never hidden
  if (Array.isArray(inc.contradictions) && inc.contradictions.length)
    h += `<div class="rep-row"><span class="rep-k">Contradictions</span><ul class="beh-list contra-list">${inc.contradictions.map(c => `<li>${esc(c)}</li>`).join('')}</ul></div>`
  // FIX 6 — per-object timeline
  if (Array.isArray(inc.timelines) && inc.timelines.length)
    h += `<div class="rep-row"><span class="rep-k">Timeline</span><div class="rep-para mono">${inc.timelines.map(t =>
      `${esc(t.object)}: first F${esc(t.first_frame)} → peak F${esc(t.peak_frame)} (${esc(t.peak_confidence)}%) → last F${esc(t.last_frame)}`).join('<br>')}</div></div>`
  // FIX 10 — why NOT each alternative verdict
  if (inc.why_not && typeof inc.why_not === 'object') {
    const order = [['confirmed_violation', 'Confirmed'], ['needs_review', 'Needs Review'], ['no_violation', 'No Violation'], ['technical_failure', 'Technical Failure']]
    const items = order.filter(([k]) => inc.why_not[k]).map(([k, lbl]) => `<li><b>Why not ${esc(lbl)}?</b> ${esc(inc.why_not[k])}</li>`).join('')
    if (items) h += `<div class="rep-row"><span class="rep-k">Decision rationale</span><ul class="beh-list">${items}</ul></div>`
  }
  return h
}

// examinerBlock shows the recorded human decision, or interactive controls to
// make one. The decision is stored ALONGSIDE the system verdict, never over it.
function examinerBlock(inc) {
  if (inc.examiner_decision) {
    return `<div class="rep-row exam-done"><span class="rep-k">Examiner Decision</span>
      <div class="rep-para"><b class="exam-tag exam-${esc(inc.examiner_decision)}">${esc(String(inc.examiner_decision).toUpperCase())}</b>${
        inc.examiner_id ? ' · ' + esc(inc.examiner_id) : ''}${
        inc.reviewed_at ? ' · ' + esc(fmtTime(inc.reviewed_at)) : ''}${
        inc.examiner_note ? `<br><i>“${esc(inc.examiner_note)}”</i>` : ''}</div></div>`
  }
  const id = esc(String(inc.id))
  return `<div class="rep-row exam-box"><span class="rep-k">Examiner Review</span>
    <input id="exam-id" class="exam-input" placeholder="examiner id (optional)">
    <textarea id="exam-note" class="exam-input exam-note" placeholder="note (optional) — e.g. 'folded paper, not a phone'"></textarea>
    <div class="exam-btns">
      <button class="exam-btn confirm" onclick="window.__submitReview('${id}','confirmed')">Confirm</button>
      <button class="exam-btn reject"  onclick="window.__submitReview('${id}','rejected')">Reject</button>
      <button class="exam-btn incon"   onclick="window.__submitReview('${id}','inconclusive')">Inconclusive</button>
      <button class="exam-btn tech"    onclick="window.__submitReview('${id}','technical')">Technical</button>
    </div></div>`
}

async function submitReview(id, decision) {
  const noteEl = document.getElementById('exam-note')
  const idEl   = document.getElementById('exam-id')
  const note        = noteEl ? noteEl.value : ''
  const examiner_id = idEl ? idEl.value : ''
  try {
    const res = await fetch(`${API_BASE}/review?id=${encodeURIComponent(id)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision, note, examiner_id }),
    })
    if (!res.ok) { alert('Review failed: ' + res.status); return }
    const data = await res.json().catch(() => ({}))
    // Optimistically reflect the decision in the open modal.
    const inc = _incidents.find(i => String(i.id) === String(id))
    if (inc) {
      inc.examiner_decision = decision
      inc.examiner_note     = note
      inc.examiner_id       = examiner_id
      inc.reviewed_at       = data.reviewed_at || new Date().toISOString()
      inc.integrity_hash    = data.integrity_hash || inc.integrity_hash
      renderModal(inc)
    }
  } catch (e) { alert('Review error: ' + (e && e.message)) }
}
window.__submitReview = submitReview

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
          <span>${statusLabel(inc)}</span>
          <span class="conf">${conf}</span>
        </div>
        ${inc.summary ? `<div class="rep-para modal-summary">${esc(inc.summary)}</div>` : ''}
        ${evidenceRows(inc)}
        <div class="rep-row"><span class="rep-k">Likelihood</span>
          <span class="rep-likelihood ${like}">${esc(like.toUpperCase())}</span></div>
        <div class="rep-row"><span class="rep-k">Materials Detected</span>
          <div class="tags">${mats.length ? mats.map(m => `<span class="tag">${esc(m)}</span>`).join('') : '<span class="tag none">None detected</span>'}</div></div>
        <div class="rep-row"><span class="rep-k">Suspicious Behaviors</span>
          ${behs.length ? `<ul class="beh-list">${behs.map(b => `<li>${esc(b)}</li>`).join('')}</ul>` : '<div class="rep-para">None flagged.</div>'}</div>
        ${inc.intruder_confirmed
          ? `<div class="rep-row"><span class="rep-k">Intruder / People</span><div class="rep-para" style="color:#dc2626;font-weight:700">${intruderText(inc)}</div></div>`
          : `<div class="rep-row"><span class="rep-k">People in Frame</span><div class="rep-para">Candidate only</div></div>`}
        ${phoneRowHtml(inc)}
        ${notesRowHtml(inc)}
        <div class="rep-row"><span class="rep-k">Trigger</span><div class="rep-para">${esc(triggerLabel(inc.trigger_reason))}</div></div>
        ${inc.gaze_analysis ? `<div class="rep-row"><span class="rep-k">Gaze Analysis</span><div class="rep-para">${esc(inc.gaze_analysis)}</div></div>` : ''}
        ${inc.visual_evidence ? `<div class="rep-row"><span class="rep-k">Visual Evidence</span><div class="rep-para">${esc(inc.visual_evidence)}</div></div>` : ''}
        ${inc.recommendation ? `<div class="rep-row"><div class="rep-reco"><b>Recommendation:</b> ${esc(inc.recommendation)}</div></div>` : ''}
        ${examinerBlock(inc)}
        ${inc.integrity_hash ? `<div class="rep-row"><span class="rep-k">Integrity</span><div class="rep-para mono seal-hash">🔒 ${esc(inc.integrity_hash)}</div></div>` : ''}
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
  // Render ONE continuous page exactly the size of the report — NO pagination, so
  // the PDF can never cut through a collage/card or truncate the bottom; it is a
  // pixel match of the on-screen preview. The iframe is pinned to (0,0) because the
  // overlay centres it and html2canvas would otherwise bake in that offset.
  const cw = doc.body.scrollWidth  || 794
  const ch = doc.body.scrollHeight || 1122
  const pxToMm = px => (px * 25.4) / 96
  // Resolution: render at up to 2.5× so text and screenshots stay razor-sharp.
  // The ONLY ceiling is the browser's real max-canvas size — we scale as high as
  // fits. The OLD formula (min(2, 10000/ch)) silently dropped BELOW 1× on a tall
  // report and rasterised the whole PDF at sub-native resolution → that was the
  // blur. Now we cap by the actual Chromium canvas side (32767 px) + area budget,
  // and never fall below native (1×).
  const MAX_SIDE = 32767
  const MAX_AREA = 268435456            // 2^28 px — safe Chromium canvas area budget
  const scale = Math.max(1, Math.min(
    2.5,                                // target crispness (2.5× device pixels)
    MAX_SIDE / ch,                      // height must fit the canvas
    MAX_SIDE / cw,                      // width must fit the canvas
    Math.sqrt(MAX_AREA / (cw * ch))     // total area must fit
  ))
  const prevStyle = iframe.style.cssText
  iframe.style.cssText = `position:fixed;left:0;top:0;width:${cw}px;height:${ch}px;border:0;z-index:99999;background:#fff`
  try {
    await html2pdf().set({
      margin:      0,
      filename:    filename,
      image:       { type: 'jpeg', quality: 0.98 },
      // pagebreak mode [] disables ALL page-break logic. Without this, html2pdf's
      // default ['css','legacy'] mode reads the report's `@page A4` rule +
      // `page-break-inside:avoid` on each card and paginates into A4 pages —
      // which pushed tall cards to the next page and left big WHITE GAPS. We want
      // exactly what the custom tall jsPDF format below promises: ONE continuous
      // page, pixel-matching the on-screen preview, no orphan gaps ever.
      pagebreak:   { mode: [] },
      html2canvas: { scale, useCORS: true, backgroundColor: '#ffffff', logging: false, width: cw, height: ch, windowWidth: cw, windowHeight: ch, x: 0, y: 0, scrollX: 0, scrollY: 0 },
      jsPDF:       { unit: 'mm', format: [pxToMm(cw), pxToMm(ch) + 10], orientation: 'portrait' }
    }).from(doc.body).save()
    return true
  } catch (e) {
    console.error('[pdf] generation failed:', e)
    alert('PDF generation failed: ' + e.message)
    return false
  } finally {
    iframe.style.cssText = prevStyle
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
    const warn = objWarn(obj, f.second_person)
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
  <span class="v-label">${statusLabel(inc)}</span>
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
    ${inc.intruder_confirmed
      ? `<div class="fld"><div class="fld-k" style="color:#dc2626">Intruder / People in Frame</div><div class="fld-v" style="color:#dc2626;font-weight:700">${intruderText(inc)}</div></div>`
      : `<div class="fld"><div class="fld-k">People in Frame</div><div class="fld-v">Candidate only</div></div>`}
    ${hasPhone(inc) ? `<div class="fld"><div class="fld-k" style="color:#dc2626">Phone Detection</div><div class="fld-v" style="color:#dc2626;font-weight:700">${phoneText(inc)}</div></div>` : ''}
    ${hasNotes(inc) ? `<div class="fld"><div class="fld-k" style="color:#dc2626">Notes / Paper</div><div class="fld-v" style="color:#dc2626;font-weight:700">📝 NOTES / PAPER DETECTED — confirmed by AI vision</div></div>` : ''}
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
  <span>Generated by <b>Proctor AI</b> · Vision analysis by Groq (primary) · Gemini (fallback)</span>
  <span>Trigger: ${e(triggerLabel(inc.trigger_reason))}</span>
</div>

</body>
</html>`

  await downloadReportPdf(html, `Proctor-Incident-${inc.id}.pdf`)
}

document.getElementById('report-pdf')?.addEventListener('click', downloadPDF)
document.getElementById('modal-pdf')?.addEventListener('click', downloadPDF)
// Official, server-rendered, tamper-sealed PDF (deterministic — the artifact of record).
document.getElementById('official-pdf-btn')?.addEventListener('click', () => {
  window.open(`${API_BASE}/session-pdf`, '_blank')
})

// ── Session Overview + Frame Timeline (Layer 1 continuous log) → PDF HTML ────
// Renders the "nothing discarded, only layered" section from the signal summary
// the browser posted at End Session. Degrades to nothing if no log was captured.
function sessionOverviewHtml(session) {
  const sig = session && session.signals
  const e = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c]))
  const aiFrames = Array.isArray(session.incidents)
    ? session.incidents.reduce((n, i) => n + (Array.isArray(i.per_frame) ? i.per_frame.length : 0), 0)
    : 0
  const triggered = Array.isArray(session.incidents) ? session.incidents.length : 0
  if (!sig) return ''

  const mmss = s => { s = Math.max(0, Number(s) || 0); return `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}` }
  const dur  = Number(sig.duration_secs) || 0
  const durStr = dur > 0 ? `${Math.floor(dur/60)}m ${dur%60}s` : '--'
  const g = sig.gaze || {}, f = sig.face || {}, m = sig.motion || {}, q = sig.quality || {}, ev = sig.events || {}

  const stat = (k, v, warn) =>
    `<div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">${e(k)}</b><span style="${warn?'color:#dc2626;font-weight:700':'font-weight:600'}">${v}</span></div>`

  const step = Number(sig.sample_secs) || 1
  const stepLabel = step >= 60 ? `${Math.round(step/60)}m` : `${step}s`
  const tl = Array.isArray(sig.timeline) ? sig.timeline : []
  const tlRows = tl.map(r => {
    const anomaly = r.phone || (Number(r.face) >= 2) || (r.gaze !== 'AHEAD' && r.gaze !== 'UNKNOWN' && r.gaze !== 'NO_FACE') || Number(r.motion) > 35
    const cell = 'border:1px solid #e2e8f0;padding:3px 7px'
    const rowBg = anomaly ? 'background:#fef2f2;color:#991b1b' : ''
    return `<tr style="${rowBg}"><td style="${cell}">${mmss(r.sec)}</td><td style="${cell}">${e(r.face)}</td><td style="${cell}">${e(r.gaze)}</td><td style="${cell}">${r.hand?'yes':'—'}</td><td style="${cell}">${e(r.motion)}</td><td style="${cell};font-weight:${r.phone?'700':'400'}">${r.phone?'YES':'—'}</td></tr>`
  }).join('')

  return `
<div style="padding:12px 20px;border-top:6px solid #4f46e5;background:#fbfbff">
  <div style="font-size:8.5pt;font-weight:700;color:#4f46e5;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">Continuous Capture — Session Overview</div>
  <div style="font-size:8.5pt;color:#94a3b8;margin-bottom:10px">Layer 1: every frame logged lightweightly @ ${e(sig.avg_fps)} FPS avg · heavy AI ran only on the ${triggered} triggered incident(s) below</div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px 16px;font-size:10pt">
    ${stat('Total Frames Captured', e(sig.total_frames))}
    ${stat('Session Duration', e(durStr))}
    ${stat('Frames Analyzed by AI', `${aiFrames} of ${e(sig.total_frames)}`)}
    ${stat('Trigger Events', e(ev.total ?? 0), (Number(ev.total)||0) > 0)}
    ${stat('Gaze — Ahead', e(g.AHEAD ?? 0))}
    ${stat('Gaze — Off-screen', e((Number(g.LEFT)||0)+(Number(g.RIGHT)||0)+(Number(g.UP)||0)+(Number(g.DOWN)||0)), true)}
    ${stat('Frames w/ 2+ Faces', e(f.multi ?? 0), (Number(f.multi)||0) > 0)}
    ${stat('Hand-Visible Frames', e(sig.hand_present_frames ?? 0))}
    ${stat('On-Device Phone Hint', `${e(sig.phone_hint_frames ?? 0)} frame(s)`, (Number(sig.phone_hint_frames)||0) > 0)}
    ${stat('Motion (avg / peak)', `${e(m.avg ?? 0)} / ${e(m.max ?? 0)}`)}
    ${stat('High-Motion Frames', e(m.high_frames ?? 0))}
    ${stat('Frame Quality (good/poor)', `${e(q.good ?? 0)} / ${e(q.poor ?? 0)}`)}
  </div>
  ${tlRows ? `
  <div style="font-size:8pt;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin:14px 0 6px">Frame Timeline — sampled every ${stepLabel} from the ~45 FPS stream (full truth, nothing discarded)</div>
  <table style="width:100%;border-collapse:collapse;font-size:8.5pt;font-family:'Consolas','Courier New',monospace">
    <thead><tr>
      <th style="background:#eef2ff;border:1px solid #e2e8f0;padding:4px 7px;text-align:left;color:#4f46e5">Time</th>
      <th style="background:#eef2ff;border:1px solid #e2e8f0;padding:4px 7px;text-align:left;color:#4f46e5">Face</th>
      <th style="background:#eef2ff;border:1px solid #e2e8f0;padding:4px 7px;text-align:left;color:#4f46e5">Gaze</th>
      <th style="background:#eef2ff;border:1px solid #e2e8f0;padding:4px 7px;text-align:left;color:#4f46e5">Hand</th>
      <th style="background:#eef2ff;border:1px solid #e2e8f0;padding:4px 7px;text-align:left;color:#4f46e5">Motion</th>
      <th style="background:#eef2ff;border:1px solid #e2e8f0;padding:4px 7px;text-align:left;color:#4f46e5">Phone</th>
    </tr></thead>
    <tbody>${tlRows}</tbody>
  </table>` : ''}
</div>`
}

// ── Session Report (End Session button) ────────────────────────────────────
async function generateSessionPDF() {
  const btn = document.getElementById('end-session-btn')
  // resetBtn also LIFTS finalization mode, so a cancelled/failed finalize
  // re-enables live detection instead of leaving the proctor frozen.
  const resetBtn = () => {
    window.__finalizing = false
    if (btn) { btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9l6 6M15 9l-6 6"/></svg> End Session'; btn.disabled = false }
  }
  if (btn) { btn.textContent = 'Finalizing…'; btn.disabled = true }

  // ── FINALIZATION PHASE ────────────────────────────────────────────────────
  // "End Session" is a REQUEST to finalize, NOT an immediate PDF build. We close
  // the door to new incidents, then WAIT for the backend to confirm every
  // incident has finished its whole pipeline (capture → upload → AI verdict)
  // before freezing a snapshot. The report is built ONLY from that verified,
  // locked snapshot — never from live/half-finished state. This is what kills
  // the "PENDING in the PDF" bug: we don't guess with a timeout, we wait for the
  // backend to say "queue_depth == 0".
  window.__finalizing = true   // main.js checks this to stop starting NEW captures

  // 1) Let any in-flight browser capture/upload finish so its incident reaches
  //    the server before we start polling the queue.
  const capWaitStart = Date.now()
  while (window.__captureBusy && Date.now() - capWaitStart < 15000) {
    await new Promise(r => setTimeout(r, 200))
  }

  // 2) WAIT-FOR-COMPLETION GATE with a HARD QUIESCENCE WINDOW (backend-verified,
  //    not time-based). Poll the cheap /session-status until the server reports
  //    queue_depth == 0 AND has stayed at 0 continuously for QUIESCENCE_MS. The
  //    sustained window closes the millisecond race: an incident that reports
  //    "done" at one instant but whose AI response is landing right then would
  //    momentarily push depth back above 0 and RESET the window — so we only
  //    freeze the snapshot once the system is genuinely idle, not merely idle for
  //    one poll. Keeps waiting while the queue drains; a hard 5-min ceiling forces
  //    through (permanently dead AI key), and the backend has already settled
  //    stragglers as FAILED — so the report is still COMPLETE-or-FAILED.
  const HARD_CAP_MS   = 300000   // absolute ceiling — never hang past 5 minutes
  const QUIESCENCE_MS = 3000     // queue must stay empty this long before we trust "done"
  const gateStart = Date.now()
  let idleSince = 0
  try {
    for (;;) {
      const r = await fetch(`${API_BASE}/session-status`, { cache: 'no-store' })
      const st = await r.json()
      const depth = Number(st?.queue_depth) || 0
      const now = Date.now()
      if (depth === 0) {
        if (idleSince === 0) idleSince = now                    // queue just went idle — start the window
        if (now - idleSince >= QUIESCENCE_MS) break             // stayed idle long enough → genuinely done
        if (btn) btn.textContent = 'Finalizing — verifying idle…'
      } else {
        idleSince = 0                                           // work reappeared → reset the quiescence window
        if (btn) btn.textContent = `Finalizing — ${depth} analyzing…`
      }
      if (now - gateStart > HARD_CAP_MS) break                  // safety valve
      await new Promise(res => setTimeout(res, 1000))           // 1s polls → the 3s window resolves promptly
    }
  } catch (_) {
    alert('Could not reach the API server — cannot finalize the session.')
    resetBtn()
    return
  }

  // 2b) POST the Layer-1 continuous frame log (the "memory buffer") to the API
  //     BEFORE we freeze the snapshot, so both the JSON report and the official
  //     server PDF can render the SESSION OVERVIEW + FRAME TIMELINE from it. Best
  //     effort — a failure here never blocks finalizing the report.
  try {
    const sig = window.__getSessionSignals && window.__getSessionSignals()
    if (sig) {
      await fetch(`${API_BASE}/session-signals`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sig),
      })
    }
  } catch (_) { /* continuous log is a bonus — don't fail finalization over it */ }

  // 3) FREEZE THE SNAPSHOT. One authoritative fetch AFTER the gate clears; this
  //    object is the single source of truth the PDF is built from. Nothing reads
  //    live state past this point, so later updates can never mutate the report.
  if (btn) btn.textContent = 'Generating…'
  let session
  try {
    const r = await fetch(`${API_BASE}/session-report`, { cache: 'no-store' })
    session = await r.json()
  } catch (_) {
    alert('Could not reach the API server.')
    resetBtn()
    return
  }
  // `session` is now the LOCKED snapshot — it is never re-fetched below this
  // line, so everything the PDF renders comes from this single frozen object.
  // The gate above guaranteed queue_depth==0, so it contains only terminal
  // (COMPLETE or FAILED) incidents — never an in-progress one.

  // Include EVERY incident that fired this session — flagged, needs-review, AND
  // cleared — so no triggered incident is ever missing from the report. Each is
  // labelled with its own verdict; a cleared one shows as "NO CHEATING".
  const allIncidents = Array.isArray(session.incidents) ? session.incidents : []
  const incidents = allIncidents.slice()

  if (incidents.length === 0) {
    alert('No incidents were recorded this session — nothing to report.')
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
  const intruderTotal = incidents.filter(i => i.intruder_confirmed === true).length
  // "Flagged" = incidents needing attention (confirmed cheating/intruder + needs-review),
  // NOT the total incident count (the report now lists every incident, cleared ones too).
  const flaggedCount = incidents.filter(i => Number(i.cheating) === 1 || i.intruder_confirmed === true || i.review_required === true).length

  const incRows = incidents.map((inc, idx) => {
    const like     = (inc.cheating_likelihood || 'analyzing').toLowerCase()
    // A confirmed on-device violation (phone/intruder) is authoritative even while
    // the LLM is still enriching — colour it RED like the per-incident PDF/web, not
    // amber "pending", so every surface agrees on the same incident.
    const violation = Number(inc.cheating) === 1 || inc.intruder_confirmed === true
    // This is a FINALIZED report — there is no "in progress" state. An incident is
    // either COMPLETE (has a verdict) or FAILED (AI vision could not analyze it
    // before the session was finalized). Any residual pending/no-verdict is shown
    // as FAILED with an explicit reason — the report NEVER shows "PENDING".
    const failed   = !violation && (inc.llm_failed === true || inc.llm_pending === true ||
                     like === 'analyzing' || like === 'unknown' || like === '')
    const cheat    = Number(inc.cheating) === 1
    const intruder = inc.intruder_confirmed === true
    const review   = !failed && !violation && inc.review_required === true
    const conf     = !failed && inc.confidence_score != null ? `${inc.confidence_score}%` : '--'
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
           <tbody>${inc.per_frame.map(f => { const o=f.object_in_hand||'none'; const w=objWarn(o,f.second_person); return `<tr><td style="border:1px solid #e2e8f0;padding:3px 6px;${w?'background:#fef2f2;color:#991b1b':''}">${e(f.frame??'')}</td><td style="border:1px solid #e2e8f0;padding:3px 6px;${w?'background:#fef2f2;color:#991b1b':''}">${e(o)}</td><td style="border:1px solid #e2e8f0;padding:3px 6px">${e(f.gaze||'--')}</td><td style="border:1px solid #e2e8f0;padding:3px 6px">${f.second_person?'YES':'—'}</td><td style="border:1px solid #e2e8f0;padding:3px 6px">${e(f.description||'')}</td></tr>` }).join('')}</tbody>
         </table></div>`
      : ''
    const likeColor   = like === 'high' ? '#dc2626' : like === 'medium' ? '#d97706' : like === 'low' ? '#16a34a' : '#94a3b8'
    // FAILED gets its own explicit label/icon — never "pending".
    const statusText  = failed ? '✗ AI ANALYSIS FAILED'
                       : (violation ? '⚠ ' : review ? '⚑ ' : '✓ ') + statusLabel(inc)
    const statusColor = violation ? '#dc2626' : failed ? '#b45309' : review ? '#d97706' : '#16a34a'
    const rowBorder   = violation ? '#fca5a5' : failed ? '#fbbf24' : review ? '#fcd34d' : '#86efac'
    return `
    <div style="border:1.5px solid ${rowBorder};border-radius:8px;padding:14px 16px;margin-bottom:14px;page-break-inside:avoid">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap">
        <span style="background:#1e293b;color:#fff;border-radius:20px;padding:2px 10px;font-size:9pt;font-weight:700">#${idx+1}</span>
        <span style="font-size:10pt;color:#64748b">${fmtTs(inc.timestamp)}</span>
        <span style="font-size:9pt;color:#94a3b8">${e(triggerLabel(inc.trigger_reason))}</span>
        <span style="margin-left:auto;font-weight:800;color:${statusColor};font-size:10.5pt">${statusText}</span>
        ${!failed ? `<span style="color:#64748b;font-size:10pt">${conf} confidence · <span style="color:${likeColor};font-weight:700">${like.toUpperCase()} likelihood</span></span>` : ''}
      </div>
      ${inc.summary ? `<div style="margin:0 0 10px;font-weight:600;color:#334155;font-size:10pt;line-height:1.45">${e(inc.summary)}</div>` : ''}
      ${img}
      ${failed ? `<div style="margin-top:10px;padding:8px 12px;background:#fff7ed;border-left:3px solid #b45309;border-radius:0 4px 4px 0;font-size:9.5pt;color:#7c2d12"><b>AI ANALYSIS FAILED</b> — the vision service could not analyze this incident before the session was finalized (free-tier rate limit or service unavailable). The 6-frame collage above is preserved; <b>manual examiner review is required</b>.</div>` : ''}
      ${intruder ? `<div style="margin-top:8px;padding:6px 12px;background:#fef2f2;border-left:3px solid #dc2626;border-radius:0 4px 4px 0;font-size:9.5pt;color:#dc2626;font-weight:700">${intruderText(inc)}</div>` : ''}
      ${hasPhone(inc) ? `<div style="margin-top:8px;padding:6px 12px;background:#fef2f2;border-left:3px solid #dc2626;border-radius:0 4px 4px 0;font-size:9.5pt;color:#dc2626;font-weight:700">${phoneText(inc)}</div>` : ''}
      ${hasNotes(inc) ? `<div style="margin-top:8px;padding:6px 12px;background:#fef2f2;border-left:3px solid #dc2626;border-radius:0 4px 4px 0;font-size:9.5pt;color:#dc2626;font-weight:700">📝 NOTES / PAPER DETECTED — confirmed by AI vision</div>` : ''}
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
body{font-family:'Segoe UI',Arial,sans-serif;font-size:11pt;color:#1e293b;background:#fff;width:794px;margin:0;}
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
    ${(Number(session.failed_count)||Number(session.pending_count)||0) > 0 ? `<div style="color:#b45309;font-weight:700">✗ ${Number(session.failed_count)||Number(session.pending_count)} incident(s) FAILED AI analysis (rate-limited) — collages preserved, manual review required</div>` : ''}
  </div>
</div>

<div style="padding:10px 20px;background:#f8fafc;border-bottom:1px solid #e2e8f0;display:grid;grid-template-columns:repeat(4,1fr);gap:10px 16px;font-size:10pt">
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Flagged Incidents</b><span style="color:${flaggedCount > 0 ? '#dc2626' : 'inherit'};font-weight:700">${flaggedCount} of ${incidents.length}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Confirmed Cheating</b><span style="color:${(session.cheating_count||0) > 0 ? '#dc2626' : '#16a34a'};font-weight:700">${session.cheating_count || 0}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Needs Review</b><span style="color:${(session.review_count||0) > 0 ? '#d97706' : 'inherit'};font-weight:700">${session.review_count || 0}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Phone Detections</b><span style="color:${incidents.filter(hasPhone).length > 0 ? '#dc2626' : 'inherit'};font-weight:700">${incidents.filter(hasPhone).length}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Notes Detections</b><span style="color:${incidents.filter(i=>hasNotes(i)).length > 0 ? '#dc2626' : 'inherit'};font-weight:700">${incidents.filter(i=>hasNotes(i)).length}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Intruder Incidents</b><span style="color:${intruderTotal > 0 ? '#dc2626' : 'inherit'};font-weight:700">${intruderTotal}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Max People in Frame</b><span style="color:${(session.max_people||1) > 1 ? '#dc2626' : 'inherit'};font-weight:700">${session.max_people ?? 1}</span></div>
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">AI Analyzed</b>${session.analyzed_count ?? session.total} / ${session.total}</div>
  ${(Number(session.failed_count)||Number(session.pending_count)||0) > 0 ? `<div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">AI Failed</b><span style="color:#b45309;font-weight:700">${Number(session.failed_count)||Number(session.pending_count)} — manual review</span></div>` : ''}
  <div><b style="color:#94a3b8;font-size:8pt;text-transform:uppercase;display:block">Total Frames</b>${(session.signals && session.signals.total_frames) ?? session.total_frames ?? '--'}</div>
</div>

${sessionOverviewHtml(session)}

<div style="padding:14px 20px">
  <div style="font-size:8.5pt;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px">Incident Timeline (${incidents.length} recorded)</div>
  ${incRows}
</div>

<div style="padding:10px 20px;border-top:1px solid #e2e8f0;display:flex;justify-content:space-between;font-size:8pt;color:#94a3b8">
  <span>Generated by <b>Proctor AI</b> · Vision analysis by Groq (primary) · Gemini (fallback)</span>
  <span>For examiner use only — confidential</span>
</div>

</body></html>`

  previewReport(html, `Proctor-Session-${sessionId}.pdf`)
  // Session is finalized and the report is built → release the webcam and freeze
  // the feed so the candidate is no longer recorded.
  try { window.__endSessionCleanup && window.__endSessionCleanup() } catch (_) {}
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
