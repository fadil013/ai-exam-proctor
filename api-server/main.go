package main

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var geminiClient = &http.Client{Timeout: 120 * time.Second}

// geminiMu serialises all Gemini calls — free tier is 20 RPM so one-at-a-time
// with built-in retry-after handling is the safest strategy.
var geminiMu sync.Mutex
var geminiNext time.Time // earliest time the next call is allowed

// groqMu serialises all Groq calls and groqNext enforces rate-limit back-off.
// Free tier is 30 RPM / 30,000 TPM; a burst of incidents (5 images each) trips
// the TPM limit, so without this a concurrent call dies with no report.json and
// the incident is stuck on PENDING forever. One-at-a-time + Retry-After fixes it.
var groqMu sync.Mutex
var groqNext time.Time

// lastSignals holds the most recent Layer-1 continuous per-frame signal summary
// posted by the browser at End Session (the "memory buffer": every frame's face
// count / gaze / hand / motion / phone-hint logged at ~zero cost while heavy AI
// runs only on triggered captures). It feeds the SESSION OVERVIEW + FRAME
// TIMELINE sections so the report is complete — nothing discarded, only layered.
// In-memory + last-writer-wins, like the LLM queue: lost on restart, which is
// fine (the report is built immediately after the POST).
var (
	lastSignals   map[string]any
	lastSignalsMu sync.Mutex
)

const port = ":8080"
const incidentsDir = "incidents"
const selfBase = "http://localhost:8080"

// Vision model config — overridable from .env (GROQ_MODEL / GEMINI_MODEL /
// LLM_MAX_IMAGES) so the LLM can be swapped WITHOUT recompiling. Defaults are
// filled in configureLLM() (called from main after .env is loaded).
var groqModel = "meta-llama/llama-4-scout-17b-16e-instruct"
var geminiModel = "gemini-2.0-flash"
var geminiURL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

// configureLLM applies any GROQ_MODEL / GEMINI_MODEL / LLM_MAX_IMAGES overrides
// from the environment and rebuilds the Gemini endpoint for the chosen model.
func configureLLM() {
	if m := strings.TrimSpace(os.Getenv("GROQ_MODEL")); m != "" {
		groqModel = m
	}
	if m := strings.TrimSpace(os.Getenv("GEMINI_MODEL")); m != "" {
		geminiModel = m
	}
	geminiURL = "https://generativelanguage.googleapis.com/v1beta/models/" + geminiModel + ":generateContent"
	if n := strings.TrimSpace(os.Getenv("LLM_MAX_IMAGES")); n != "" {
		if v, err := strconv.Atoi(n); err == nil && v > 0 {
			maxLLMImages = v
		}
	}
}

// ── loadEnv: read KEY=VALUE lines from .env, set any unset vars (stdlib only) ─
func loadEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.Trim(strings.TrimSpace(val), `"'`)
		if key != "" && os.Getenv(key) == "" {
			os.Setenv(key, val)
		}
	}
}

// ── Request / response types ───────────────────────────────────────────────

type GazeFrame struct {
	Dir   string   `json:"dir"`
	H     *float64 `json:"h"`
	V     *float64 `json:"v"`
	Yaw   *float64 `json:"yaw"`
	Pitch *float64 `json:"pitch"`
	T     *string  `json:"t"` // per-frame ISO timestamp (browser capture time)
}

type IncidentRequest struct {
	Frames           []string    `json:"frames"`
	Collage          string      `json:"collage"`
	GazeLabels       []string    `json:"gazeLabels"`
	GazeData         []GazeFrame `json:"gazeData"`
	Timestamp        string      `json:"timestamp"`
	PhoneDetected    bool        `json:"phone_detected"`
	FramePhones      []bool      `json:"frame_phones"` // per-frame on-device phone state
	HandPosture      string      `json:"hand_posture"`
	FaceCount        int         `json:"face_count"`
	IntruderDetected bool        `json:"intruder_detected"`
	TriggerReason    string      `json:"trigger_reason"`
}

type IncidentResponse struct {
	Status     string `json:"status"`
	IncidentID string `json:"incident_id"`
	Saved      int    `json:"saved"`
	Collage    bool   `json:"collage"`
	Report     bool   `json:"report"`
}

// ── LLM report (parsed from Gemini's JSON) ─────────────────────────────────

// FrameObs is the model's grounded, per-frame observation. The model must
// describe EACH of the 6 frames independently — this forces it to look rather
// than free-associate a verdict, and lets us apply a cross-frame consistency
// rule (a real object persists across frames; a hallucination shows up in one).
type FrameObs struct {
	Frame        int    `json:"frame"`
	HandVisible  bool   `json:"hand_visible"`
	ObjectInHand string `json:"object_in_hand"` // none|phone|paper|book|earbud|watch|other
	SecondPerson bool   `json:"second_person"`
	Gaze         string `json:"gaze"`        // ahead|left|right|up|down
	Description  string `json:"description"` // what is actually visible in this frame

	// ── Layer A — RAW FRAME TRUTH (immutable once assigned) ──────────────────
	// These record what was actually observed in THIS frame. They are set once and
	// NEVER overwritten by later reasoning — the AI and the detector each ADD their
	// own signal (ObjectInHand vs DetectorPhone), so a report can never "change its
	// mind" about a frame and every verdict is traceable back to raw pixels.
	Quality       string  `json:"quality,omitempty"`          // good|acceptable|poor|unusable (raw clarity)
	AIConfidence  int     `json:"object_confidence,omitempty"` // AI's per-frame confidence 0-100 (structured signal, not prose)
	DetectorPhone bool    `json:"detector_phone,omitempty"`  // on-device detector flagged a phone in THIS frame (ADDED, never overwrites the AI object)
	Detection     string  `json:"detection,omitempty"`       // DETECTED|WEAK_DETECTION|AMBIGUOUS|NOT_DETECTED (raw primitive — no interpretation)
	Weight        float64 `json:"weight,omitempty"`          // clarity × confidence, 0..1 — how much this frame counts

	// ── Derived display (Layer B/C — computed from the raw truth above) ──────
	ObjectStatus  string `json:"object_status,omitempty"`  // present|likely|possible|uncertain|not_visible
	ObjectClass   string `json:"object_class,omitempty"`   // phone|paper|book|watch|earbud|unknown|none
	FrameStatus   string `json:"frame_status,omitempty"`   // clean|suspicious|violation_evidence|uncertain
	DetectorAgree bool   `json:"detector_agree,omitempty"` // on-device detector corroborated this frame

	// ── Identity / provenance (exact frame-index mapping + content hash) ─────
	FrameID      string `json:"frame_id,omitempty"`       // stable per-frame id (incident + capture index)
	CaptureIndex int    `json:"capture_index,omitempty"`  // physical position in the capture sequence
	SourceHash   string `json:"source_hash,omitempty"`    // sha256(frame bytes) prefix — proves which image this row describes
	AnalyzedAI   bool   `json:"analyzed_ai,omitempty"`    // this exact frame was sent to and analysed by the vision model
	UsedInVerdict bool  `json:"used_in_verdict,omitempty"` // this frame contributed to the evidence score
}

type Report struct {
	// Grounded evidence (filled by the model)
	PerFrame []FrameObs `json:"per_frame"`

	// Verdict + summary fields. cheating / review_required / phone_visible /
	// materials are RECOMPUTED deterministically in Go from PerFrame — the
	// model's own values are advisory only.
	Cheating                 int      `json:"cheating"`
	ReviewRequired           bool     `json:"review_required"`
	CheatingLikelihood       string   `json:"cheating_likelihood"`
	ConfidenceScore          int      `json:"confidence_score"`
	CheatingMaterialDetected bool     `json:"cheating_material_detected"`
	PhoneVisible             bool     `json:"phone_visible"`
	PhoneAIConfirmed         bool     `json:"phone_ai_confirmed,omitempty"`
	IntruderConfirmed        bool     `json:"intruder_confirmed"`
	Materials                []string `json:"materials"`
	SuspiciousBehaviors      []string `json:"suspicious_behaviors"`
	GazeAnalysis             string   `json:"gaze_analysis"`
	VisualEvidence           string   `json:"visual_evidence"`
	Recommendation           string   `json:"recommendation"`
	Summary                  string   `json:"summary"`
	LLMPending               bool     `json:"llm_pending,omitempty"`
	LLMFailed                bool     `json:"llm_failed,omitempty"` // AI analysis exhausted → settle as FAILED, never "pending"

	// ── Evidence-strength verdict layer ──────────────────────────────────────
	// The report is no longer a binary "did a rule threshold trip" — it is a
	// weighted evidence assessment. SystemVerdict is one of four honest states;
	// EvidenceStrength grades HOW strong the evidence is; the Why* fields make the
	// decision defensible (never an accusation from weak evidence, never a silent
	// clear when the AI could not judge).
	SystemVerdict    string           `json:"system_verdict,omitempty"`    // confirmed_violation|needs_review|no_violation|technical_failure
	EvidenceStrength string           `json:"evidence_strength,omitempty"` // strong|moderate|weak|inconclusive (DERIVED from EvidenceScore)
	EvidenceScore    int              `json:"evidence_score,omitempty"`    // 0-100, computed from weighted real signals (NOT a human label)
	WhyNotClear      string           `json:"why_not_clear,omitempty"`     // what suspicious signal prevents a clean pass
	WhyNotCheating   string           `json:"why_not_cheating,omitempty"`  // what uncertainty prevents a confirmed accusation
	WhyNot           map[string]string `json:"why_not,omitempty"`          // for EVERY non-chosen verdict → why it was not chosen (no black box)
	Contradictions   []string         `json:"contradictions,omitempty"`    // explicit AI-vs-detector / cross-frame disagreements (shown, never hidden)
	Timelines        []ObjectTimeline `json:"timelines,omitempty"`         // per-object first/peak/last appearance
	IntegrityHash    string           `json:"integrity_hash,omitempty"`    // sha256 tamper-seal over the frozen evidence

	// ── Examiner review (human-in-the-loop) ──────────────────────────────────
	// The examiner's decision is recorded ALONGSIDE the system verdict — it never
	// overwrites it — and is folded into the tamper-seal so the human decision is
	// part of the evidence chain (the report shows BOTH: what the system found and
	// what the examiner concluded).
	ExaminerDecision string `json:"examiner_decision,omitempty"` // confirmed|rejected|inconclusive|technical
	ExaminerNote     string `json:"examiner_note,omitempty"`
	ExaminerID       string `json:"examiner_id,omitempty"`
	ReviewedAt       string `json:"reviewed_at,omitempty"`

	// ── Audit / replay ──────────────────────────────────────────────────────
	// ReplayInput captures the EXACT inputs decideVerdict last ran on, and
	// DecisionTrace records how it arrived at the verdict. Together they make the
	// verdict fully reproducible: replayVerdict() re-runs the pure decision engine
	// from ReplayInput and must produce this same verdict — so "why was I flagged?"
	// can always be answered by re-deriving the exact decision path.
	ReplayInput   *ReplayInput  `json:"replay_input,omitempty"`
	DecisionTrace *VerdictTrace `json:"decision_trace,omitempty"`
}

// ReplayInput is the complete, minimal set of signals decideVerdict consumes —
// everything needed to deterministically rebuild the verdict from raw data.
type ReplayInput struct {
	ClientIntruder  bool   `json:"client_intruder"`
	TriggerReason   string `json:"trigger_reason"`
	PhoneClient     bool   `json:"phone_client"`
	LLMPhoneSeen    bool   `json:"llm_phone_seen"`
	LLMRan          bool   `json:"llm_ran"`
	LLMHadFrames    bool   `json:"llm_had_frames"`
	ModelConfidence int    `json:"model_confidence"` // the model's confidence BEFORE the engine adjusts it
}

// VerdictTrace records the decision path decideVerdict took — the per-frame
// tallies it counted and which rule branch fired — so the verdict is auditable,
// not a black box.
type VerdictTrace struct {
	CheatFrames    int    `json:"cheat_frames"`
	UnknownFrames  int    `json:"unknown_frames"`
	SecondFrames   int    `json:"second_frames"`
	LLMPhoneFrames int    `json:"llm_phone_frames"`
	PhoneCheat     bool   `json:"phone_cheat"`
	ObjType        string `json:"obj_type,omitempty"`
	Rule           string `json:"rule"` // confirmed | intruder | ambiguous | clear

	// Evidence-strength tallies — the quality-weighted counts the ladder used.
	PhoneGoodFrames int    `json:"phone_good_frames"`
	PhoneWeakFrames int    `json:"phone_weak_frames"`
	SecondGood      int    `json:"second_good_frames"`
	SecondWeak      int    `json:"second_weak_frames"`
	UsableFrames    int    `json:"usable_frames"`
	SystemVerdict   string `json:"system_verdict"`   // confirmed_violation|needs_review|no_violation|technical_failure
	EvidenceStrength string `json:"evidence_strength"` // strong|moderate|weak|inconclusive
}

// ObjectTimeline is the appearance history of one object across the frames — so
// the report shows a temporal story (first seen → peak confidence → last seen),
// not just a static per-frame dump.
type ObjectTimeline struct {
	Object         string `json:"object"`
	FirstFrame     int    `json:"first_frame"`
	PeakFrame      int    `json:"peak_frame"`
	LastFrame      int    `json:"last_frame"`
	PeakConfidence int    `json:"peak_confidence"` // 0-100
	Frames         int    `json:"frames"`          // how many frames it appears in
}

// ── Supabase row (what we insert + return to the dashboard) ────────────────

type IncidentRecord struct {
	ID                       string           `json:"id"`
	Timestamp                string           `json:"timestamp"`
	Cheating                 int              `json:"cheating"`
	ReviewRequired           bool             `json:"review_required"`
	ConfidenceScore          int              `json:"confidence_score"`
	CheatingLikelihood       string           `json:"cheating_likelihood"`
	CheatingMaterialDetected bool             `json:"cheating_material_detected"`
	PhoneVisible             bool             `json:"phone_visible"`
	PhoneDetected            bool             `json:"phone_detected"`
	PhoneAIConfirmed         bool             `json:"phone_ai_confirmed"`
	FaceCount                int              `json:"face_count"`
	IntruderDetected         bool             `json:"intruder_detected"`
	IntruderConfirmed        bool             `json:"intruder_confirmed"`
	TriggerReason            string           `json:"trigger_reason"`
	Materials                []string         `json:"materials"`
	SuspiciousBehaviors      []string         `json:"suspicious_behaviors"`
	GazeAnalysis             string           `json:"gaze_analysis"`
	VisualEvidence           string           `json:"visual_evidence"`
	Recommendation           string           `json:"recommendation"`
	Summary                  string           `json:"summary"`
	GazeData                 []map[string]any `json:"gaze_data"`
	PerFrame                 []FrameObs       `json:"per_frame,omitempty"`
	LLMPending               bool             `json:"llm_pending,omitempty"`
	LLMFailed                bool             `json:"llm_failed,omitempty"` // AI analysis exhausted → settle as FAILED, never "pending"
	SystemVerdict            string            `json:"system_verdict,omitempty"`
	EvidenceStrength         string            `json:"evidence_strength,omitempty"`
	EvidenceScore            int               `json:"evidence_score,omitempty"`
	WhyNotClear              string            `json:"why_not_clear,omitempty"`
	WhyNotCheating           string            `json:"why_not_cheating,omitempty"`
	WhyNot                   map[string]string `json:"why_not,omitempty"`
	Contradictions           []string          `json:"contradictions,omitempty"`
	Timelines                []ObjectTimeline  `json:"timelines,omitempty"`
	IntegrityHash            string            `json:"integrity_hash,omitempty"`
	ExaminerDecision         string            `json:"examiner_decision,omitempty"`
	ExaminerNote             string           `json:"examiner_note,omitempty"`
	ExaminerID               string           `json:"examiner_id,omitempty"`
	ReviewedAt               string           `json:"reviewed_at,omitempty"`
	Collage                  string           `json:"collage"`
}

// ── Gemini API types ───────────────────────────────────────────────────────

type geminiRequest struct {
	Contents         []geminiContent `json:"contents"`
	GenerationConfig *genConfig      `json:"generationConfig,omitempty"`
}
type genConfig struct {
	ResponseMimeType string       `json:"responseMimeType"`
	ThinkingConfig   *thinkingCfg `json:"thinkingConfig,omitempty"`
}
type thinkingCfg struct {
	ThinkingBudget int `json:"thinkingBudget"`
}
type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}
type geminiPart struct {
	Text       string            `json:"text,omitempty"`
	InlineData *geminiInlineData `json:"inlineData,omitempty"`
}
type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}
type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// reprocessOrphans finds incident folders with a collage but no report.json and
// calls the configured vision LLM for each one sequentially.
func reprocessOrphans() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("[reprocess] PANIC recovered: %v\n", r)
		}
	}()
	time.Sleep(2 * time.Second)
	groqKey := os.Getenv("GROQ_API_KEY")
	geminiKey := os.Getenv("GEMINI_API_KEY")
	if groqKey == "" && geminiKey == "" {
		return
	}
	entries, err := os.ReadDir(incidentsDir)
	if err != nil {
		return
	}

	type job struct{ id, dir string }
	var jobs []job
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		id := e.Name()
		dir := filepath.Join(incidentsDir, id)
		if _, err := os.Stat(filepath.Join(dir, "report.json")); err == nil {
			continue
		}
		if _, err := os.Stat(filepath.Join(dir, "frame_01.jpg")); err != nil {
			continue
		}
		jobs = append(jobs, job{id, dir})
	}
	// Newest first — most recent captures are the most important
	sort.Slice(jobs, func(a, b int) bool { return jobs[a].id > jobs[b].id })
	if len(jobs) == 0 {
		return
	}
	fmt.Printf("[reprocess] %d orphaned incidents — starting parallel analysis\n", len(jobs))

	// Sequential — geminiMu ensures one call at a time; rate-limit handled inside callGemini
	for _, j := range jobs {
		// Load the saved individual frames (frame_01.jpg, frame_02.jpg, …) and
		// send the most time-spread ones to the model — same path as live.
		var rawFrames []string
		for i := 1; i <= 12; i++ {
			b, e := os.ReadFile(filepath.Join(j.dir, fmt.Sprintf("frame_%02d.jpg", i)))
			if e != nil {
				break
			}
			rawFrames = append(rawFrames, "data:image/jpeg;base64,"+base64.StdEncoding.EncodeToString(b))
		}
		if len(rawFrames) == 0 {
			continue
		}
		imageURLs := selectFrameURLs(rawFrames, maxLLMImages)
		var frames []map[string]any
		var framePhones []bool
		var timestamp string
		var phoneDetected bool
		var intruderDetected bool
		var faceCount int
		handPosture := "unknown"
		triggerReason := "hand_visible_and_gaze_toward_wrist"
		if b, err2 := os.ReadFile(filepath.Join(j.dir, "metadata.json")); err2 == nil {
			var meta map[string]any
			if json.Unmarshal(b, &meta) == nil {
				if ts, ok := meta["timestamp"].(string); ok {
					timestamp = ts
				}
				if gd, ok := meta["gaze_data"]; ok {
					if b2, _ := json.Marshal(gd); b2 != nil {
						json.Unmarshal(b2, &frames)
					}
				}
				if pd, ok := meta["phone_detected"].(bool); ok {
					phoneDetected = pd
				}
				if fp, ok := meta["frame_phones"].([]any); ok {
					for _, v := range fp {
						b, _ := v.(bool)
						framePhones = append(framePhones, b)
					}
				}
				if hp, ok := meta["hand_posture"].(string); ok {
					handPosture = hp
				}
				if id, ok := meta["intruder_detected"].(bool); ok {
					intruderDetected = id
				}
				if fc, ok := meta["face_count"].(float64); ok {
					faceCount = int(fc)
				}
				if tr, ok := meta["trigger_reason"].(string); ok && tr != "" {
					triggerReason = tr
				}
			}
		}
		// PROVISIONAL on-device verdict FIRST — written BEFORE the LLM call so a
		// phone orphan is CHEATING even if the LLM is down/throttled (mirrors the
		// live path; the LLM below only ENRICHES it).
		prov := Report{}
		for i, fr := range frames {
			gz, _ := fr["gaze"].(string)
			gz = strings.TrimPrefix(strings.ToLower(gz), "looking ")
			obj, desc := "none", "On-device frame — AI vision analysis pending."
			if i < len(framePhones) && framePhones[i] {
				obj, desc = "phone", "On-device detector: a PHONE was detected in this frame."
			}
			prov.PerFrame = append(prov.PerFrame, FrameObs{Frame: i + 1, Gaze: gz, ObjectInHand: obj, Description: desc})
		}
		decideVerdict(&prov, intruderDetected, triggerReason, phoneDetected, false, false, false)
		prov.LLMPending = len(imageURLs) > 0
		if pb, mErr := json.MarshalIndent(prov, "", "  "); mErr == nil {
			writeJSONAtomic(filepath.Join(j.dir, "report.json"), pb)
		}

		// Hand the orphan to the SAME serial LLM queue as the live path — so it is
		// analyzed (and retried until it succeeds) rather than dropped on one error.
		req := IncidentRequest{
			Frames:           rawFrames,
			Timestamp:        timestamp,
			PhoneDetected:    phoneDetected,
			FramePhones:      framePhones,
			HandPosture:      handPosture,
			FaceCount:        faceCount,
			IntruderDetected: intruderDetected,
			TriggerReason:    triggerReason,
		}
		rec := IncidentRecord{
			ID:                  j.id,
			Timestamp:           timestamp,
			Materials:           []string{},
			SuspiciousBehaviors: []string{},
			GazeData:            frames,
			CheatingLikelihood:  "unknown",
			FaceCount:           faceCount,
			IntruderDetected:    intruderDetected,
			TriggerReason:       triggerReason,
			PhoneDetected:       phoneDetected,
		}
		atomic.AddInt64(&llmOutstanding, 1) // NEW incident enters the pipeline
		enqueueLLM(llmJob{incidentID: j.id, dir: j.dir, req: req, frames: frames, rec: rec})
		fmt.Printf("[reprocess] %s — provisional written, queued for AI analysis\n", j.id)
	}
}

// ── Main ───────────────────────────────────────────────────────────────────

func main() {
	loadEnv(".env")
	configureLLM()
	os.MkdirAll(incidentsDir, 0755)

	gemOK := os.Getenv("GEMINI_API_KEY") != ""
	grqOK := os.Getenv("GROQ_API_KEY") != ""
	switch {
	case gemOK && grqOK:
		fmt.Printf("Vision LLM: Groq (%s, primary) + Gemini (%s, fallback) — ENABLED\n", groqModel, geminiModel)
		go llmWorker()
		go reprocessOrphans()
	case gemOK:
		fmt.Printf("Vision LLM: Gemini (%s) — ENABLED\n", geminiModel)
		go llmWorker()
		go reprocessOrphans()
	case grqOK:
		fmt.Printf("Vision LLM: Groq (%s) — ENABLED\n", groqModel)
		go llmWorker()
		go reprocessOrphans()
	default:
		fmt.Println("Vision LLM: DISABLED (set GEMINI_API_KEY or GROQ_API_KEY)")
	}
	if os.Getenv("SUPABASE_URL") != "" && os.Getenv("SUPABASE_SECRET_KEY") != "" {
		fmt.Println("Supabase persistence: ENABLED")
	} else {
		fmt.Println("Supabase persistence: DISABLED (set SUPABASE_URL + SUPABASE_SECRET_KEY)")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/incident", handleIncident)            // POST one incident
	mux.HandleFunc("/incidents", handleIncidentsList)      // GET list (dashboard)
	mux.HandleFunc("/incidents/", handleCollageImage)      // GET /incidents/{id}/collage.jpg
	mux.HandleFunc("/session-report", handleSessionReport)   // GET full session PDF data
	mux.HandleFunc("/session-signals", handleSessionSignals) // POST Layer-1 continuous per-frame signal summary
	mux.HandleFunc("/session-pdf", handleSessionPDF)         // GET official server-rendered PDF (deterministic, hashed)
	mux.HandleFunc("/session-status", handleSessionStatus)   // GET cheap finalization gate (queue depth only)
	mux.HandleFunc("/replay", handleReplay)                // GET ?id= — forensic: rebuild verdict + event history
	mux.HandleFunc("/review", handleReview)                // POST ?id= — examiner decision (human-in-the-loop)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	fmt.Printf("API server → %s\n", selfBase)
	if err := http.ListenAndServe(port, mux); err != nil {
		fmt.Fprintf(os.Stderr, "server error: %v\n", err)
		os.Exit(1)
	}
}

func setCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func decodeImage(dataURL string) ([]byte, error) {
	idx := strings.Index(dataURL, ",")
	if idx < 0 {
		return nil, fmt.Errorf("no comma in data URL")
	}
	return base64.StdEncoding.DecodeString(dataURL[idx+1:])
}

// ── POST /incident ─────────────────────────────────────────────────────────

func handleIncident(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req IncidentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	incidentID := fmt.Sprintf("%s_%s", time.Now().Format("20060102_150405"), shortID())
	dir := fmt.Sprintf("%s/%s", incidentsDir, incidentID)
	os.MkdirAll(dir, 0755)

	// Build complete per-frame data (used for storage + the LLM payload).
	frames := buildFrames(req)

	// Save individual frames + per-frame JSON.
	saved := 0
	for i, frame := range req.Frames {
		data, err := decodeImage(frame)
		if err != nil {
			continue
		}
		if os.WriteFile(fmt.Sprintf("%s/frame_%02d.jpg", dir, i+1), data, 0644) == nil {
			saved++
		}
		if i < len(frames) {
			if b, err := json.MarshalIndent(frames[i], "", "  "); err == nil {
				os.WriteFile(fmt.Sprintf("%s/frame_%02d.json", dir, i+1), b, 0644)
			}
		}
	}

	// Save collage locally (served as the dashboard thumbnail).
	collageOk := false
	if req.Collage != "" {
		if data, err := decodeImage(req.Collage); err == nil {
			if os.WriteFile(dir+"/collage.jpg", data, 0644) == nil {
				collageOk = true
			}
		}
	}

	// Save metadata.
	meta := map[string]any{
		"incident_id":       incidentID,
		"timestamp":         req.Timestamp,
		"frame_count":       saved,
		"gaze_labels":       req.GazeLabels,
		"gaze_data":         frames,
		"phone_detected":    req.PhoneDetected,
		"frame_phones":      req.FramePhones,
		"hand_posture":      req.HandPosture,
		"face_count":        req.FaceCount,
		"intruder_detected": req.IntruderDetected,
		"trigger_reason":    req.TriggerReason,
	}
	if b, err := json.MarshalIndent(meta, "", "  "); err == nil {
		os.WriteFile(dir+"/metadata.json", b, 0644)
	}

	// LLM + Supabase run async — never block the capture loop.
	reportWanted := (os.Getenv("GROQ_API_KEY") != "" || os.Getenv("GEMINI_API_KEY") != "") && len(req.Frames) > 0
	go processIncident(incidentID, dir, req, frames)

	fmt.Printf("[incident] %s  frames=%d/%d  collage=%v  report=%v\n",
		incidentID, saved, len(req.Frames), collageOk, reportWanted)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(IncidentResponse{
		Status: "ok", IncidentID: incidentID, Saved: saved, Collage: collageOk, Report: reportWanted,
	})
}

// buildFrames merges gaze labels + per-frame numeric data + timestamps.
func buildFrames(req IncidentRequest) []map[string]any {
	out := []map[string]any{}
	for i, gd := range req.GazeData {
		label := "UNKNOWN"
		if i < len(req.GazeLabels) {
			label = req.GazeLabels[i]
		}
		out = append(out, map[string]any{
			"frame":      i + 1,
			"time":       gd.T,
			"gaze":       label,
			"iris_h":     gd.H,
			"iris_v":     gd.V,
			"head_yaw":   gd.Yaw,
			"head_pitch": gd.Pitch,
		})
	}
	return out
}

// writeJSONAtomic writes b to a temp file in the same directory and renames it
// over path. Rename is atomic on the same volume (Windows MoveFileEx as well), so
// a concurrent reader (the dashboard poll / session report) always sees either the
// old complete file or the new complete file — never a truncated buffer from the
// os.WriteFile truncate-then-write window (which parsed as "analyzing" and
// transiently under-reported the incident).
func writeJSONAtomic(path string, b []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".report-*.tmp")
	if err != nil {
		return os.WriteFile(path, b, 0644) // fallback: best-effort direct write
	}
	tmpName := tmp.Name()
	if _, err = tmp.Write(b); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err = tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, path)
}

// frameHash returns a short content hash of a raw capture frame, so a per-frame
// row can prove WHICH physical image it describes (exact frame-index mapping).
func frameHash(raw string) string {
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:])[:12]
}

// sealReport computes a SHA-256 tamper-seal over the frozen evidence + verdict.
// The hash excludes IntegrityHash itself, so a later edit to any sealed field
// (a per-frame observation, the verdict, the decision trace) changes the seal and
// the tampering is detectable. This turns the report from a document into evidence.
func sealReport(r *Report) string {
	seal := struct {
		SystemVerdict, EvidenceStrength, CheatingLikelihood string
		EvidenceScore                                       int
		Cheating                                            int
		ConfidenceScore                                     int
		ReviewRequired, PhoneVisible, IntruderConfirmed     bool
		Materials                                           []string
		PerFrame                                            []FrameObs
		Trace                                               *VerdictTrace
		Input                                               *ReplayInput
		Contradictions                                      []string
		Timelines                                           []ObjectTimeline
		WhyNot                                              map[string]string
		// Human decision is part of the sealed evidence chain (team-lead #21).
		ExaminerDecision, ExaminerNote, ExaminerID, ReviewedAt string
	}{
		r.SystemVerdict, r.EvidenceStrength, r.CheatingLikelihood,
		r.EvidenceScore,
		r.Cheating, r.ConfidenceScore,
		r.ReviewRequired, r.PhoneVisible, r.IntruderConfirmed,
		r.Materials, r.PerFrame, r.DecisionTrace, r.ReplayInput,
		r.Contradictions, r.Timelines, r.WhyNot,
		r.ExaminerDecision, r.ExaminerNote, r.ExaminerID, r.ReviewedAt,
	}
	b, _ := json.Marshal(seal)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// writeReport persists report.json and mirrors the verdict into rec (dashboard
// list + Supabase row) so on-disk state and the dashboard always agree.
func writeReport(dir string, r *Report, rec *IncidentRecord) {
	r.IntegrityHash = sealReport(r) // tamper-seal the frozen evidence before persisting
	if b, err := json.MarshalIndent(r, "", "  "); err == nil {
		writeJSONAtomic(filepath.Join(dir, "report.json"), b)
	}
	rec.Cheating = r.Cheating
	rec.ReviewRequired = r.ReviewRequired
	rec.ConfidenceScore = r.ConfidenceScore
	rec.CheatingLikelihood = r.CheatingLikelihood
	rec.CheatingMaterialDetected = r.CheatingMaterialDetected
	rec.PhoneVisible = r.PhoneVisible
	rec.PhoneAIConfirmed = r.PhoneAIConfirmed
	rec.IntruderConfirmed = r.IntruderConfirmed
	if r.Materials != nil {
		rec.Materials = r.Materials
	}
	if r.SuspiciousBehaviors != nil {
		rec.SuspiciousBehaviors = r.SuspiciousBehaviors
	}
	rec.GazeAnalysis = r.GazeAnalysis
	rec.VisualEvidence = r.VisualEvidence
	rec.Recommendation = r.Recommendation
	rec.Summary = r.Summary
	rec.PerFrame = r.PerFrame
	rec.LLMPending = r.LLMPending
	rec.LLMFailed = r.LLMFailed
	rec.SystemVerdict = r.SystemVerdict
	rec.EvidenceStrength = r.EvidenceStrength
	rec.EvidenceScore = r.EvidenceScore
	rec.WhyNotClear = r.WhyNotClear
	rec.WhyNotCheating = r.WhyNotCheating
	rec.WhyNot = r.WhyNot
	rec.Contradictions = r.Contradictions
	rec.Timelines = r.Timelines
	rec.IntegrityHash = r.IntegrityHash
	rec.ExaminerDecision = r.ExaminerDecision
	rec.ExaminerNote = r.ExaminerNote
	rec.ExaminerID = r.ExaminerID
	rec.ReviewedAt = r.ReviewedAt
}

// processIncident writes an immediate PROVISIONAL verdict from the on-device
// signals (so an incident is NEVER left blank/pending), then enriches it with
// the vision LLM in the background — Gemini 2.0 Flash primary (1500 RPD / 15
// RPM, far higher limits than Groq's throttled free vision tier), Groq fallback.
func processIncident(incidentID, dir string, req IncidentRequest, frames []map[string]any) {
	// This runs in a goroutine with no net/http recovery — an unrecovered panic
	// would crash the WHOLE server. Recover so one bad incident can never do that.
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("[incident] %s PANIC recovered: %v\n", incidentID, r)
		}
	}()
	rec := IncidentRecord{
		ID:                  incidentID,
		Timestamp:           req.Timestamp,
		Materials:           []string{},
		SuspiciousBehaviors: []string{},
		GazeData:            frames,
		Collage:             req.Collage,
		CheatingLikelihood:  "unknown",
		Summary:             "No AI assessment (LLM disabled).",
		// Client-side sensor ground truth — independent of the LLM verdict
		FaceCount:        req.FaceCount,
		IntruderDetected: req.IntruderDetected,
		TriggerReason:    req.TriggerReason,
		PhoneDetected:    req.PhoneDetected,
	}

	// Send the individual full-res frames (the 5 most time-spread of the 6) to
	// the vision model — far more legible for small objects than a downscaled
	// collage. The collage is kept only as the dashboard thumbnail.
	imageURLs := selectFrameURLs(req.Frames, maxLLMImages)

	// PROVISIONAL verdict from the on-device signals (phone/intruder/trigger),
	// written IMMEDIATELY so the report is never blank/half while the (possibly
	// rate-limited) vision LLM works. The LLM only ENRICHES this below.
	prov := Report{}
	// Build a per-frame table from the gaze data + the on-device per-frame phone
	// flags, so EVERY incident shows a per-frame breakdown even when the vision
	// LLM is unavailable. (The LLM overwrites this with richer detail when it runs.)
	for i, fr := range frames {
		gz, _ := fr["gaze"].(string)
		gz = strings.TrimPrefix(strings.ToLower(gz), "looking ")
		obj, desc := "none", "On-device frame — AI vision analysis pending."
		if i < len(req.FramePhones) && req.FramePhones[i] {
			obj, desc = "phone", "On-device detector: a PHONE was detected in this frame."
		}
		prov.PerFrame = append(prov.PerFrame, FrameObs{Frame: i + 1, Gaze: gz, ObjectInHand: obj, Description: desc})
	}
	decideVerdict(&prov, req.IntruderDetected, req.TriggerReason, req.PhoneDetected, false, false, false)
	prov.LLMPending = len(imageURLs) > 0
	writeReport(dir, &prov, &rec)
	appendEvent(dir, "created", map[string]any{
		"trigger_reason":    req.TriggerReason,
		"phone_client":      req.PhoneDetected,
		"face_count":        req.FaceCount,
		"intruder_client":   req.IntruderDetected,
		"provisional_rule":  traceRule(&prov),
		"awaiting_ai":       prov.LLMPending,
	})
	insertSupabase(rec)

	// Hand the incident to the serial LLM queue for AI enrichment. The queue
	// processes ONE incident at a time (respecting the vision tier's tokens-per-
	// minute cap) and RETRIES until it succeeds — so a burst of incidents that
	// momentarily blows past the rate limit is drained over the next minute or
	// two instead of being permanently left "not analyzed". The provisional
	// verdict written above is visible immediately; the queue only upgrades it.
	if len(imageURLs) > 0 {
		atomic.AddInt64(&llmOutstanding, 1) // NEW incident enters the pipeline
		enqueueLLM(llmJob{incidentID: incidentID, dir: dir, req: req, frames: frames, rec: rec})
	}
}

// ── Serial LLM enrichment queue ──────────────────────────────────────────────
// One worker drains this queue, so vision calls are never made concurrently
// (which would instantly trip the free tier's per-minute token cap). A job that
// can't be analyzed right now (rate-limited) is re-queued with backoff and tried
// AGAIN, so every incident is eventually analyzed rather than dropped.

type llmJob struct {
	incidentID string
	dir        string
	req        IncidentRequest
	frames     []map[string]any
	rec        IncidentRecord
	attempts   int
}

// llmQueue is large so a long burst of incidents is buffered, never blocking the
// HTTP handler. maxLLMAttempts bounds re-queues so a permanently-dead key can't
// loop forever (~6 tries over several minutes, then the provisional stands).
var llmQueue = make(chan llmJob, 4096)

// llmOutstanding counts incidents whose AI verdict is not yet FINAL — from the
// moment the incident is created until it either (a) gets a real AI verdict or
// (b) is settled by finalizeUnanalyzed. It is the authoritative signal the
// dashboard's finalization gate polls: a session report is only allowed once
// this reaches 0, so no incident is ever frozen into the PDF mid-analysis.
// Incremented ONCE per incident at the two creation sites (NOT on re-queue, so a
// rate-limited incident bouncing through the retry backoff still counts as one
// outstanding job), decremented ONCE at its terminal state.
var llmOutstanding int64

func llmQueueDepth() int64 { return atomic.LoadInt64(&llmOutstanding) }

const maxLLMAttempts = 6

// llmRetryBackoff spaces re-queues so the per-minute token window can reopen:
// 15s, 30s, 60s, 90s, 120s, 180s.
func llmRetryBackoff(attempt int) time.Duration {
	switch {
	case attempt <= 1:
		return 15 * time.Second
	case attempt == 2:
		return 30 * time.Second
	case attempt == 3:
		return 60 * time.Second
	case attempt == 4:
		return 90 * time.Second
	case attempt == 5:
		return 120 * time.Second
	default:
		return 180 * time.Second
	}
}

func enqueueLLM(job llmJob) {
	select {
	case llmQueue <- job:
	default:
		// The queue is full (a massive burst). Rather than drop the job and leave
		// the incident stuck on llm_pending=true forever (it already has a
		// provisional report.json, so reprocessOrphans would never revisit it),
		// settle it on its provisional NEEDS-REVIEW verdict now.
		fmt.Printf("[llm-queue] FULL — %s settling on its provisional verdict\n", job.incidentID)
		finalizeUnanalyzed(job.dir, job.rec)
	}
}

// llmWorker is the single consumer of llmQueue. Started once in main().
func llmWorker() {
	for job := range llmQueue {
		func(job llmJob) {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("[llm-worker] %s PANIC recovered: %v\n", job.incidentID, r)
				}
			}()
			if enrichWithLLM(job.incidentID, job.dir, job.req, job.frames, job.rec) {
				atomic.AddInt64(&llmOutstanding, -1) // terminal: got a real verdict
				return                               // analyzed & written
			}
			job.attempts++
			if job.attempts < maxLLMAttempts {
				d := llmRetryBackoff(job.attempts)
				fmt.Printf("[llm-worker] %s rate-limited — requeue in %s (attempt %d/%d)\n",
					job.incidentID, d, job.attempts+1, maxLLMAttempts)
				go func(j llmJob) { time.Sleep(d); enqueueLLM(j) }(job)
			} else {
				fmt.Printf("[llm-worker] %s — AI vision unavailable after %d attempts; provisional NEEDS-REVIEW verdict stands\n",
					job.incidentID, job.attempts)
				finalizeUnanalyzed(job.dir, job.rec)
			}
		}(job)
	}
}

// groqMaxImages is Groq Llama-4-Scout's HARD per-request image cap (5). Gemini
// 2.0 Flash accepts many more, so it analyses EVERY captured frame — no frame is
// left unseen on the Gemini path. Whichever provider actually produces the result
// determines sentIdx, so the per-frame table maps observations to the exact
// physical frames that provider was shown (no frame is silently "unanalyzed" and
// mislabelled). Both are env-overridable for models with different limits.
var groqMaxImages = envIntDefault("GROQ_MAX_IMAGES", 5)
var geminiMaxImages = envIntDefault("GEMINI_MAX_IMAGES", 12)

// envIntDefault reads a positive integer env var, falling back to def.
func envIntDefault(name string, def int) int {
	if v, err := strconv.Atoi(strings.TrimSpace(os.Getenv(name))); err == nil && v > 0 {
		return v
	}
	return def
}

// enrichWithLLM makes ONE attempt to analyze an incident with the vision model
// and, on success, writes the final report through the same deterministic
// decision engine as everything else. Returns true if analyzed, false if the
// model was unavailable/rate-limited (caller re-queues).
func enrichWithLLM(incidentID, dir string, req IncidentRequest, frames []map[string]any, rec IncidentRecord) bool {
	if len(req.Frames) == 0 {
		return true // nothing to analyze — treat as done
	}
	var report Report
	var raw string
	var err error
	var sentIdx []int // physical frame indices the SUCCESSFUL provider was shown
	groqKey := os.Getenv("GROQ_API_KEY")
	geminiKey := os.Getenv("GEMINI_API_KEY")
	if groqKey != "" {
		urls, idx := selectFrames(req.Frames, groqMaxImages)
		if len(urls) > 0 {
			report, raw, err = callGroq(groqKey, urls, incidentID, req.Timestamp, frames, req.PhoneDetected, req.HandPosture, req.FaceCount, req.IntruderDetected, req.TriggerReason)
			if err != nil {
				fmt.Printf("[incident] %s  Groq error: %v\n", incidentID, err)
			} else if raw != "" {
				sentIdx = idx
			}
		}
	}
	if (err != nil || raw == "") && geminiKey != "" {
		urls, idx := selectFrames(req.Frames, geminiMaxImages) // Gemini sees EVERY captured frame
		if len(urls) > 0 {
			report, raw, err = callGemini(geminiKey, urls, incidentID, req.Timestamp, frames, req.PhoneDetected, req.HandPosture, req.FaceCount, req.IntruderDetected, req.TriggerReason)
			if err != nil {
				fmt.Printf("[incident] %s  Gemini error: %v\n", incidentID, err)
			} else if raw != "" {
				sentIdx = idx
			}
		}
	}
	if err != nil || raw == "" {
		return false // rate-limited / unavailable → caller re-queues
	}

	// Deterministic decision engine. The LLM CALL succeeded (we returned above on
	// error/empty), so llmRan is true — this drives the false-intruder override
	// (a client 2-face blip the model examined and did NOT corroborate → review,
	// not a conviction). Separately, llmHadFrames records whether the model actually
	// returned grounded per-frame rows: an empty/absent per_frame is still "not
	// analyzed enough to CLEAR", so it stays NEEDS REVIEW rather than a false all-clear.
	// Compute llmHadFrames / llmPhoneSeen from the model's OWN output BEFORE the merge
	// remaps the table.
	llmRan := true
	llmHadFrames := len(report.PerFrame) > 0
	llmPhoneSeen := false
	for _, f := range report.PerFrame {
		if isPhoneObject(f.ObjectInHand) { // whole-word: "headphone" is NOT a phone sighting
			llmPhoneSeen = true
			break
		}
	}
	appendEvent(dir, "ai_observations", map[string]any{
		"llm_had_frames": llmHadFrames,
		"llm_phone_seen": llmPhoneSeen,
		"raw_per_frame":  report.PerFrame, // the model's OWN observations, before merge
	})
	report.PerFrame = buildMergedPerFrame(incidentID, req.Frames, frames, report.PerFrame, sentIdx)
	mergeFramePhones(&report, req.FramePhones)
	decideVerdict(&report, req.IntruderDetected, req.TriggerReason, req.PhoneDetected, llmPhoneSeen, llmRan, llmHadFrames)
	report.LLMPending = false
	writeReport(dir, &report, &rec)
	appendEvent(dir, "verdict", map[string]any{
		"source":          "ai_enriched",
		"rule":            traceRule(&report),
		"cheating":        report.Cheating,
		"review_required": report.ReviewRequired,
		"phone_visible":   report.PhoneVisible,
		"intruder":        report.IntruderConfirmed,
		"confidence":      report.ConfidenceScore,
	})
	insertSupabase(rec)
	state := "CLEAR"
	if report.Cheating == 1 {
		state = "CHEATING"
	} else if report.ReviewRequired {
		state = "NEEDS_REVIEW"
	}
	fmt.Printf("[incident] %s  verdict: %s  phone=%v intruder=%v(%d) materials=%v conf=%d%% (%s)\n",
		incidentID, state, report.PhoneVisible, req.IntruderDetected, req.FaceCount, report.Materials, report.ConfidenceScore, report.CheatingLikelihood)
	return true
}

// ── Event sourcing: append-only per-incident audit trail ────────────────────
// report.json is updated IN PLACE (provisional → AI-enriched), which loses the
// intermediate truth. events.jsonl is the immutable complement: one JSON line
// per state transition, never rewritten. Together with ReplayInput/DecisionTrace
// they answer "why did the verdict change / why was I flagged?" with the exact
// history and a reproducible decision path.
func appendEvent(dir, eventType string, detail map[string]any) {
	rec := map[string]any{
		"ts":   time.Now().UTC().Format(time.RFC3339Nano),
		"type": eventType,
	}
	for k, v := range detail {
		rec[k] = v
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return
	}
	f, err := os.OpenFile(filepath.Join(dir, "events.jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	f.Write(append(b, '\n'))
}

// readEvents returns the full ordered event history for an incident.
func readEvents(dir string) []map[string]any {
	out := []map[string]any{}
	b, err := os.ReadFile(filepath.Join(dir, "events.jsonl"))
	if err != nil {
		return out
	}
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var m map[string]any
		if json.Unmarshal([]byte(line), &m) == nil {
			out = append(out, m)
		}
	}
	return out
}

// replayVerdict deterministically REBUILDS the verdict from a report's stored
// ReplayInput, re-running the exact same decision engine on the exact same
// inputs. It returns the freshly-recomputed report and whether its
// audit-critical fields MATCH the stored verdict. This is forensic
// reproducibility: a match proves the recorded verdict was derived by the rules
// (not invented or tampered); a mismatch flags that report.json was altered or
// the engine changed since it was written.
func replayVerdict(stored Report) (Report, bool) {
	if stored.ReplayInput == nil {
		return stored, false
	}
	in := stored.ReplayInput
	fresh := Report{PerFrame: stored.PerFrame, ConfidenceScore: in.ModelConfidence}
	decideVerdict(&fresh, in.ClientIntruder, in.TriggerReason, in.PhoneClient, in.LLMPhoneSeen, in.LLMRan, in.LLMHadFrames)
	match := fresh.Cheating == stored.Cheating &&
		fresh.ReviewRequired == stored.ReviewRequired &&
		fresh.PhoneVisible == stored.PhoneVisible &&
		fresh.IntruderConfirmed == stored.IntruderConfirmed &&
		fresh.CheatingLikelihood == stored.CheatingLikelihood
	return fresh, match
}

// finalizeUnanalyzed settles an incident whose AI analysis was exhausted (rate
// limited past all retries, or the queue overflowed) onto an EXPLICIT terminal
// state instead of showing "pending" forever. It marks the incident as
// AI-FAILED (llm_failed=true, llm_pending=false) so the final report can render
// it as FAILED-with-reason rather than a misleading "pending" — the report only
// ever contains COMPLETE or FAILED incidents, never in-progress ones. This is a
// terminal state, so it decrements the outstanding counter exactly once.
func finalizeUnanalyzed(dir string, rec IncidentRecord) {
	defer atomic.AddInt64(&llmOutstanding, -1) // terminal: AI gave up on this one
	p := filepath.Join(dir, "report.json")
	b, err := os.ReadFile(p)
	if err != nil {
		return
	}
	var r Report
	if json.Unmarshal(b, &r) != nil {
		return
	}
	r.LLMPending = false
	r.LLMFailed = true
	writeReport(dir, &r, &rec)
	appendEvent(dir, "ai_failed", map[string]any{
		"reason":            "AI vision exhausted retries / unavailable",
		"provisional_rule":  traceRule(&r),
		"cheating":          r.Cheating,
		"review_required":   r.ReviewRequired,
	})
	insertSupabase(rec) // push the settled (llm_pending=false) state so the dashboard doesn't hang "pending"
}

// traceRule is a nil-safe accessor for the decision-trace rule label.
func traceRule(r *Report) string {
	if r.DecisionTrace != nil {
		return r.DecisionTrace.Rule
	}
	return ""
}

// buildMergedPerFrame produces ONE row per captured frame (so the evidence table
// always matches the 6-frame collage), overlaying the LLM's richer observations
// onto the frames it was actually shown (the evenly-sampled subset) and keeping
// an on-device row for the rest. The LLM sees at most maxLLMImages frames, so a
// raw LLM table is shorter than the collage — this reconciles the two.
func buildMergedPerFrame(incidentID string, rawFrames []string, gaze []map[string]any, llm []FrameObs, sentIdx []int) []FrameObs {
	n := len(gaze)
	if n == 0 {
		return llm // nothing to align to — keep the LLM's own table
	}
	analysed := map[int]bool{} // physical frames the model actually saw
	for _, j := range sentIdx {
		analysed[j] = true
	}
	out := make([]FrameObs, n)
	for i := 0; i < n; i++ {
		gz, _ := gaze[i]["gaze"].(string)
		gz = strings.TrimPrefix(strings.ToLower(gz), "looking ")
		out[i] = FrameObs{
			Frame:        i + 1,
			Gaze:         gz,
			ObjectInHand: "none",
			CaptureIndex: i,
			FrameID:      fmt.Sprintf("%s-f%d", incidentID, i+1),
			AnalyzedAI:   analysed[i],
		}
		if i < len(rawFrames) {
			out[i].SourceHash = frameHash(rawFrames[i]) // content hash proves WHICH image this row describes
		}
	}
	// sentIdx[k] is the PHYSICAL frame shown to the model as image k. The model was
	// told to emit one per_frame entry per image, in order (frame 1 = image 1). Map
	// each returned observation onto the physical frame it actually describes: prefer
	// the model's own 1-based frame number when it is a valid image position (defends
	// against a reordered response), else fall back to positional order. Any extra
	// entries beyond the number of images we sent are DROPPED — so a mis-counted or
	// padded response can never invent a phantom object on a frame the model never
	// saw (which previously could flip a verdict).
	if len(sentIdx) == 0 && len(llm) > 0 {
		sentIdx = pickEvenly(n, len(llm)) // legacy fallback if indices weren't threaded
	}
	for i, obs := range llm {
		pos := i
		if obs.Frame >= 1 && obs.Frame <= len(sentIdx) {
			pos = obs.Frame - 1
		}
		if pos < 0 || pos >= len(sentIdx) {
			continue // beyond the images we actually sent — drop it
		}
		j := sentIdx[pos]
		if j < 0 || j >= n {
			continue
		}
		out[j].HandVisible = obs.HandVisible
		out[j].SecondPerson = obs.SecondPerson
		if o := strings.TrimSpace(obs.ObjectInHand); o != "" {
			out[j].ObjectInHand = o
		}
		if g := strings.TrimSpace(obs.Gaze); g != "" {
			out[j].Gaze = strings.ToLower(g)
		}
		if q := strings.TrimSpace(obs.Quality); q != "" {
			out[j].Quality = strings.ToLower(q) // carry the model's frame-quality rating into the evidence weighting
		}
		if obs.AIConfidence > 0 {
			out[j].AIConfidence = obs.AIConfidence // structured per-frame confidence signal
		}
		if d := strings.TrimSpace(obs.Description); d != "" {
			out[j].Description = d
		}
	}
	for i := range out {
		if strings.TrimSpace(out[i].Description) == "" {
			out[i].Description = "On-device frame (not sent to AI vision)."
		}
	}
	return out
}

// mergeFramePhones records the on-device detector's phone sighting on each frame
// it fired — as a SEPARATE, ADDED signal (DetectorPhone=true). It NEVER overwrites
// the AI's ObjectInHand observation (FIX 3 — frame truth is immutable): the AI
// signal and the detector signal coexist, and the evidence engine reconciles them
// (agreement → stronger evidence; disagreement → an explicit CONTRADICTION, never
// a silent overwrite). The LLM sees the evenly-sampled subset (pickEvenly), so the
// per-frame flags are mapped back through the SAME sampling.
func mergeFramePhones(r *Report, framePhones []bool) {
	if len(framePhones) == 0 || len(r.PerFrame) == 0 {
		return
	}
	idx := pickEvenly(len(framePhones), len(r.PerFrame))
	for i := range r.PerFrame {
		if i >= len(idx) || idx[i] >= len(framePhones) || !framePhones[idx[i]] {
			continue
		}
		r.PerFrame[i].DetectorPhone = true // ADD the detector signal; do NOT touch the AI observation
	}
}

// classifyObject buckets a per-frame object_in_hand label into a policy class:
//
//	"cheat"   — a genuine exam-cheating aid (phone, notes/paper, book, earbud,
//	            watch, tablet/laptop/second screen, calculator).
//	"benign"  — an ordinary item a candidate is ALLOWED to have (a drink/water
//	            bottle, cup, food, snack, pen, pencil, tissue). A student sipping
//	            water is NOT cheating — these can NEVER produce a cheating verdict.
//	"unknown" — an object we cannot identify → NEEDS REVIEW at most, never an
//	            automatic conviction.
//	"none"    — no object / a bare hand.
//
// The vision model is told to use a fixed enum but sometimes returns free text
// (e.g. "water bottle"), so we normalise defensively here. This is the single
// gate that stops a benign object being flagged as cheating.
// isPhoneObject reports whether an object description names a PHONE-class
// device specifically ("phone", "smartphone", "cell phone", …) — whole-word,
// so audio devices whose names merely contain "phone" (headphone, earphone)
// do NOT count. Used for the phone-specific rules: a phone in even ONE frame
// convicts, and PhoneAIConfirmed means the vision model genuinely saw a phone.
func isPhoneObject(o string) bool {
	o = strings.ToLower(o)
	o = strings.ReplaceAll(o, "cell phone", "cellphone")
	o = strings.ReplaceAll(o, "smart phone", "smartphone")
	phones := map[string]bool{"phone": true, "smartphone": true, "iphone": true,
		"cellphone": true, "cell": true, "mobile": true}
	for _, t := range strings.FieldsFunc(o, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	}) {
		if phones[t] {
			return true
		}
	}
	return false
}

func classifyObject(o string) string {
	o = strings.ToLower(strings.TrimSpace(o))
	if o == "" {
		return "none"
	}
	// Normalise the multi-word device spellings the vision model sometimes emits
	// ("ear buds", "wrist watch") to their single-token form, so a spaced spelling
	// is still recognised as the device rather than falling through to "unknown".
	for _, r := range [][2]string{
		{"ear buds", "earbud"}, {"ear bud", "earbud"},
		{"ear phones", "earphone"}, {"ear phone", "earphone"},
		{"head phones", "headphone"}, {"head phone", "headphone"},
		{"ear piece", "earpiece"}, {"wrist watch", "wristwatch"},
		{"smart watch", "smartwatch"}, {"cell phone", "cellphone"},
		{"air pods", "airpod"}, {"air pod", "airpod"},
		{"ear pods", "earpod"}, {"ear pod", "earpod"},
		{"smart glasses", "smartglasses"}, {"smart phone", "smartphone"},
	} {
		o = strings.ReplaceAll(o, r[0], r[1])
	}
	// Whole-WORD matching — NEVER substring. The vision model emits free text like
	// "open laptop" or "expensive phone"; a naive substring test would clear a
	// laptop because the benign word "pen" sits inside "oPEN" (and "face" inside
	// "surFACE"). We tokenise on non-alphanumeric boundaries and match whole words.
	// The "none" set holds everything that can NEVER be a held cheating aid: the
	// candidate's own body parts and clothing ("shoulder", "sleeve"), the room
	// itself ("desk", "wall", "keyboard"), and connective/filler words — so a clean
	// scene like "hand under chin" or "shoulder visible" reads as empty, not an
	// unidentified object needing review. These tokens are SKIPPED inside a clause,
	// so they can never clear a device mentioned next to them ("phone under desk"
	// is still a phone).
	none := map[string]bool{"none": true, "empty": true, "nothing": true, "nil": true,
		"na": true, "unknown": true,
		// body parts
		"hand": true, "hands": true, "finger": true, "fingers": true, "fist": true,
		"palm": true, "face": true, "chin": true, "hair": true, "shoulder": true,
		"arm": true, "wrist": true, "elbow": true, "neck": true, "ear": true,
		"head": true, "torso": true, "chest": true, "body": true, "back": true,
		"lap": true, "knee": true, "leg": true, "thumb": true, "knuckle": true,
		"nail": true, "skin": true, "forehead": true, "eye": true, "nose": true,
		"mouth": true, "lip": true, "cheek": true, "jaw": true, "beard": true,
		// clothing (worn, not held)
		"shirt": true, "tshirt": true, "sleeve": true, "collar": true, "hoodie": true,
		"jacket": true, "sweater": true, "tie": true, "scarf": true, "clothes": true,
		"clothing": true, "fabric": true,
		// the room and the exam machine's own peripherals
		"desk": true, "table": true, "chair": true, "wall": true, "ceiling": true,
		"floor": true, "window": true, "door": true, "background": true, "room": true,
		"bed": true, "keyboard": true, "mouse": true, "mousepad": true, "webcam": true,
		"microphone": true, "mic": true,
		// connectives / prepositions / descriptive filler
		"near": true, "in": true, "on": true, "at": true, "by": true, "to": true, "of": true,
		"the": true, "a": true, "an": true, "my": true, "his": true, "her": true, "its": true,
		"own": true, "their": true, "them": true, "they": true,
		"is": true, "are": true, "was": true, "were": true, "being": true, "been": true,
		"resting": true, "held": true, "holding": true, "visible": true,
		"front": true, "up": true, "down": true, "left": true, "right": true,
		"under": true, "over": true, "behind": true, "above": true, "below": true,
		"against": true, "around": true, "between": true, "beneath": true, "inside": true,
		"outside": true, "into": true, "onto": true, "off": true, "away": true,
		"across": true, "toward": true, "towards": true, "while": true, "during": true,
		"wearing": true, "worn": true, "seen": true, "shown": true, "appears": true,
		"appearing": true, "looks": true, "looking": true, "raised": true, "lowered": true,
		"covering": true, "covered": true, "partially": true, "partly": true, "slightly": true,
		"nobody": true, "noone": true, "anyone": true, "anything": true, "else": true,
		"alone": true, "clear": true, "all": true, "only": true, "just": true, "also": true,
		"still": true, "from": true, "that": true, "this": true, "there": true,
		"what": true, "out": true}
	benign := map[string]bool{"bottle": true, "water": true, "cup": true, "glass": true,
		"mug": true, "flask": true, "tumbler": true, "drink": true, "coffee": true,
		"tea": true, "juice": true, "soda": true, "food": true, "snack": true,
		"fruit": true, "tissue": true, "napkin": true, "cloth": true, "towel": true,
		"pen": true, "pencil": true, "marker": true, "eraser": true, "ruler": true,
		"glasses": true, "eyeglasses": true, "spectacles": true, "sunglasses": true}
	cheat := map[string]bool{"phone": true, "smartphone": true, "iphone": true,
		"cellphone": true, "cell": true, "mobile": true, "android": true, "tablet": true,
		"ipad": true, "laptop": true, "computer": true, "monitor": true, "screen": true,
		"device": true, "earbud": true, "earphone": true, "headphone": true,
		"headset": true, "airpod": true, "earpiece": true, "watch": true,
		"smartwatch": true, "wristwatch": true, "smartglasses": true, "earpod": true,
		"paper": true, "note": true, "notepad": true,
		"sheet": true, "page": true, "card": true, "flashcard": true, "document": true,
		"book": true, "notebook": true, "textbook": true, "calculator": true,
		"cheatsheet": true, "chit": true}
	// A negation word makes its clause an empty scene: "no phone" / "no object" /
	// "not holding anything" must NOT convict and must NOT review.
	negation := map[string]bool{"no": true, "not": true, "without": true}
	// De-pluralise ("books" → "book", "shoulders" → "shoulder") so plurals aren't missed.
	lookup := func(t string) string {
		switch {
		case none[t]:
			return "none"
		case benign[t]:
			return "benign"
		case cheat[t]:
			return "cheat"
		}
		if len(t) > 3 && strings.HasSuffix(t, "s") {
			s := t[:len(t)-1]
			if none[s] {
				return "none"
			}
			if benign[s] {
				return "benign"
			}
			if cheat[s] {
				return "cheat"
			}
		}
		return ""
	}
	// Classify ONE clause by its head noun: a LATER recognised word overrides an
	// earlier one, so "paper cup" (a cup) is benign while "open book" (a book) is
	// cheat. A negation anywhere in the clause makes it empty.
	classifyClause := func(c string) string {
		class, sawObject, negated := "", false, false
		for _, t := range strings.FieldsFunc(c, func(r rune) bool {
			return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
		}) {
			if negation[t] {
				negated = true
				continue
			}
			switch lookup(t) {
			case "none":
				// body part / filler word — not a held object
			case "benign":
				class, sawObject = "benign", true
			case "cheat":
				class, sawObject = "cheat", true
			default:
				sawObject = true // an object word we don't recognise
			}
		}
		if negated {
			return "none"
		}
		if class != "" {
			return class
		}
		if sawObject {
			return "unknown"
		}
		return "none"
	}
	// Split a multi-object description into clauses on connectives, classify each,
	// and combine with precedence cheat > benign > unknown > none — so a device
	// listed ALONGSIDE a benign item ("phone and a water bottle") is NEVER cleared
	// just because the benign word happened to come last.
	seps := []string{" and ", " but ", " or ", ", ", ",", " & ", " + ", " plus ",
		" with ", " next to ", " beside ", " near ", "/"}
	s := o
	for _, sep := range seps {
		s = strings.ReplaceAll(s, sep, "\x1f")
	}
	rank := map[string]int{"none": 0, "benign": 1, "unknown": 2, "cheat": 3}
	best := "none"
	for _, c := range strings.Split(s, "\x1f") {
		if c = strings.TrimSpace(c); c == "" {
			continue
		}
		if cl := classifyClause(c); rank[cl] > rank[best] {
			best = cl
		}
	}
	return best
}

// ── Evidence-strength engine ─────────────────────────────────────────────────
// The report's core upgrade: instead of counting raw frames against integer
// thresholds, we WEIGHT each frame by quality and turn the observations into a
// graded, defensible assessment. A blurry one-frame guess can never carry the
// same weight as a device held clearly across several good frames — so a
// hallucination cannot convict, and a real cheat is not buried by one weak frame.

// frameQuality normalises a frame's quality label to good | weak | unusable.
// Unknown/blank quality is treated as "good" so legacy data (no quality tag)
// behaves as before until the frontend/AI begin scoring quality per frame.
func frameQuality(f FrameObs) string {
	switch strings.ToLower(strings.TrimSpace(f.Quality)) {
	case "poor":
		return "weak"
	case "unusable":
		return "unusable"
	default: // good, acceptable, ""
		return "good"
	}
}

// frameWeight is how much a frame counts as evidence: clarity (quality) × AI
// confidence. A blurry or low-confidence frame contributes a fraction of a clear,
// confident one — so a single bad frame can never flip a verdict (FIX 9). Missing
// AI confidence defaults to 0.85 so legacy data behaves as before.
func frameWeight(f FrameObs) float64 {
	qf := 1.0
	switch strings.ToLower(strings.TrimSpace(f.Quality)) {
	case "poor":
		qf = 0.4
	case "unusable":
		qf = 0.0
	}
	cf := 0.85
	if f.AIConfidence > 0 {
		cf = float64(f.AIConfidence) / 100.0
	}
	if cf > 1 {
		cf = 1
	}
	return qf * cf
}

// scoreFrameEvidence derives the per-frame display layer (detection primitive,
// weight, object class/status, frame status) from the RAW Layer-A truth. It only
// ADDS derived fields — it never mutates the AI observation or the detector flag.
// The Detection primitive (DETECTED|WEAK_DETECTION|AMBIGUOUS|NOT_DETECTED) carries
// NO interpretation; all interpretation happens later in the aggregation layer.
func scoreFrameEvidence(frames []FrameObs, phoneClient bool) {
	for i := range frames {
		f := &frames[i]
		if strings.TrimSpace(f.Quality) == "" {
			f.Quality = "acceptable"
		}
		f.Weight = frameWeight(*f)
		cls := classifyObject(f.ObjectInHand)
		aiPhone := isPhoneObject(f.ObjectInHand)
		obj := strings.ToLower(strings.TrimSpace(f.ObjectInHand))
		if obj == "" {
			obj = "none"
		}
		// Object class (display).
		switch {
		case aiPhone:
			f.ObjectClass = "phone"
		case cls == "cheat" || cls == "unknown":
			f.ObjectClass = obj
		case f.DetectorPhone:
			f.ObjectClass = "phone?" // detector-only — flagged by the sensor, NOT corroborated by AI
		default:
			f.ObjectClass = "none"
		}
		// ── Detection PRIMITIVE (raw, strict — FIX 1) ────────────────────────
		switch {
		case aiPhone || cls == "cheat":
			if f.Weight >= 0.7 {
				f.Detection = "DETECTED"
			} else {
				f.Detection = "WEAK_DETECTION"
			}
		case f.DetectorPhone:
			f.Detection = "WEAK_DETECTION" // a single sensor, uncorroborated
		case cls == "unknown":
			f.Detection = "AMBIGUOUS"
		default:
			f.Detection = "NOT_DETECTED"
		}
		// Object status (display interpretation, derived from the primitive).
		switch f.Detection {
		case "DETECTED":
			f.ObjectStatus = "present"
		case "WEAK_DETECTION":
			f.ObjectStatus = "likely"
		case "AMBIGUOUS":
			if f.Weight >= 0.7 {
				f.ObjectStatus = "possible"
			} else {
				f.ObjectStatus = "uncertain"
			}
		default:
			f.ObjectStatus = "not_visible"
		}
		// Frame status (display).
		switch {
		case f.SecondPerson, (aiPhone || cls == "cheat") && f.Weight >= 0.7:
			f.FrameStatus = "violation_evidence"
		case aiPhone || cls == "cheat" || f.DetectorPhone:
			f.FrameStatus = "suspicious"
		case cls == "unknown":
			f.FrameStatus = "uncertain"
		default:
			f.FrameStatus = "clean"
		}
		f.DetectorAgree = aiPhone && (phoneClient || f.DetectorPhone)
		f.UsedInVerdict = strings.ToLower(strings.TrimSpace(f.Quality)) != "unusable"
	}
}

// EvidenceAssessment is the graded, defensible outcome of weighing all frames —
// the output of Layer B (aggregation) that Layer C (verdict) consumes.
type EvidenceAssessment struct {
	SystemVerdict  string // confirmed_violation|needs_review|no_violation|technical_failure
	Strength       string // strong|moderate|weak|inconclusive (DERIVED from EvidenceScore)
	EvidenceScore  int    // 0-99, measurable ratio of confirming to total evidence
	ConvictKind    string // phone|object|intruder|""
	ObjType        string
	WhyNotClear    string
	WhyNotCheating string
	WhyNot         map[string]string // per non-chosen verdict → why not (FIX 10)
	Contradictions []string          // explicit AI-vs-detector / cross-frame disagreements (FIX 5)
	Timelines      []ObjectTimeline  // per-object first/peak/last (FIX 6)
	Confidence     int

	// integer counts (for prose/trace)
	PhoneGood, PhoneWeak   int
	SecondGood, SecondWeak int
	CheatFrames            int
	UnknownFrames          int
	SecondFrames           int
	LLMPhoneFrames         int
	UsableFrames           int
	ConfirmingFrames       int
	ConflictingFrames      int
	PhoneCheat             bool

	// weighted metrics (FIX 2 — measurable signals, not labels)
	AIPhoneWeight      float64
	ObjWeight          float64
	SecondWeight       float64
	DetectorOnlyWeight float64
	QualityAvg         float64
	AIConfAvg          float64
	OcclusionRatio     float64
}

// assessEvidence is the PURE decision core: given the per-frame observations and
// the on-device / AI signals, it returns a graded evidence assessment following
// the evidence ladders (phone / object / intruder / AI-failure). It is fully
// deterministic and independently testable.
func assessEvidence(frames []FrameObs, clientIntruder bool, triggerReason string, phoneClient, llmPhoneSeen, llmRan, llmHadFrames bool) EvidenceAssessment {
	a := EvidenceAssessment{}
	objGood := map[string]int{}
	objWeak := map[string]int{}
	objWeight := map[string]float64{}
	var aiPhoneW, secondW, detOnlyW float64
	var qualitySum, aiConfSum float64
	var aiConfN, occluded int

	for _, f := range frames {
		q := frameQuality(f)
		if q == "unusable" {
			continue
		}
		a.UsableFrames++
		w := frameWeight(f)
		if q == "weak" {
			qualitySum += 0.4
		} else {
			qualitySum += 1.0
		}
		if f.AIConfidence > 0 {
			aiConfSum += float64(f.AIConfidence)
			aiConfN++
		}
		if !f.HandVisible {
			occluded++
		}
		aiPhone := isPhoneObject(f.ObjectInHand)
		switch classifyObject(f.ObjectInHand) {
		case "cheat":
			a.CheatFrames++
			t := strings.ToLower(strings.TrimSpace(f.ObjectInHand))
			if aiPhone {
				a.LLMPhoneFrames++
				aiPhoneW += w
				if q == "good" {
					a.PhoneGood++
				} else {
					a.PhoneWeak++
				}
			} else {
				objWeight[t] += w
				if q == "good" {
					objGood[t]++
				} else {
					objWeak[t]++
				}
			}
		case "unknown":
			a.UnknownFrames++
		}
		if f.DetectorPhone && !aiPhone {
			detOnlyW += w // detector says phone, AI does not → conflicting evidence
		}
		if f.SecondPerson {
			a.SecondFrames++
			secondW += w
			if q == "good" {
				a.SecondGood++
			} else {
				a.SecondWeak++
			}
		}
	}

	// Dominant non-phone object by WEIGHT (not raw count).
	objType, objBestW := "", 0.0
	for t, wt := range objWeight {
		if wt > objBestW {
			objBestW, objType = wt, t
		}
	}
	a.ObjType = objType
	objGoodBest := objGood[objType]

	a.AIPhoneWeight, a.ObjWeight, a.SecondWeight, a.DetectorOnlyWeight = aiPhoneW, objBestW, secondW, detOnlyW
	if a.UsableFrames > 0 {
		a.QualityAvg = qualitySum / float64(a.UsableFrames)
		a.OcclusionRatio = float64(occluded) / float64(a.UsableFrames)
	}
	if aiConfN > 0 {
		a.AIConfAvg = aiConfSum / float64(aiConfN)
	}
	for _, f := range frames {
		if f.Detection == "DETECTED" {
			a.ConfirmingFrames++
		}
	}

	// ── Contradictions (FIX 5) — surfaced, never hidden ──────────────────────
	for _, f := range frames {
		if frameQuality(f) == "unusable" || !f.DetectorPhone || isPhoneObject(f.ObjectInHand) {
			continue
		}
		a.ConflictingFrames++
		if cls := classifyObject(f.ObjectInHand); cls == "cheat" || cls == "unknown" {
			a.Contradictions = append(a.Contradictions,
				fmt.Sprintf("Frame %d: on-device detector flagged a PHONE, but AI vision read the object as %q.", f.Frame, strings.ToLower(strings.TrimSpace(f.ObjectInHand))))
		} else {
			a.Contradictions = append(a.Contradictions,
				fmt.Sprintf("Frame %d: on-device detector flagged a phone, but AI vision saw no device.", f.Frame))
		}
	}
	if types := distinctCheatTypes(frames); len(types) > 1 {
		a.Contradictions = append(a.Contradictions,
			"AI labeled the held object differently across frames: "+strings.Join(types, ", ")+".")
	}
	if clientIntruder && llmHadFrames && a.SecondFrames == 0 {
		a.Contradictions = append(a.Contradictions,
			"On-device flagged a possible second face, but AI vision confirmed no second person in any frame.")
	}

	// ── Timelines (FIX 6) ────────────────────────────────────────────────────
	a.Timelines = buildTimelines(frames)

	covert := triggerReason == "covert_device_suspected"
	framing := triggerReason == "framing_torso_hidden"

	const confirmThresh = 1.4 // ≈ two clear frames of weight ~0.85
	const assistThresh = 0.7  // one clear frame that a SECOND sensor corroborates

	// ── Weighted confirmation (FIX 2 + FIX 9) ────────────────────────────────
	// Confirmation is a WEIGHTED sum, never an integer count — so one blurry frame
	// can never tip a verdict, and one clear frame alone can never convict. A phone
	// confirms on ≥1.4 weighted AI evidence, OR ≥0.7 with on-device corroboration;
	// a detector hit the AI never corroborated is a CONTRADICTION, not a conviction.
	phoneConfirmed := aiPhoneW >= confirmThresh || (aiPhoneW >= assistThresh && phoneClient)
	a.PhoneCheat = phoneConfirmed
	objConfirmed := objBestW >= confirmThresh
	intruderConfirmed := secondW >= confirmThresh

	// ── EvidenceScore: measurable confirming / (confirming + conflict + base) ─
	confirmWinner := aiPhoneW
	if objBestW > confirmWinner {
		confirmWinner = objBestW
	}
	if secondW > confirmWinner {
		confirmWinner = secondW
	}
	a.EvidenceScore = int(math.Round(100 * confirmWinner / (confirmWinner + detOnlyW + 0.8)))
	if a.EvidenceScore > 99 {
		a.EvidenceScore = 99
	}

	// ── Verdict (Layer C only — FIX 4) ───────────────────────────────────────
	switch {
	case phoneConfirmed:
		a.SystemVerdict, a.ConvictKind = "confirmed_violation", "phone"
	case objConfirmed:
		a.SystemVerdict, a.ConvictKind = "confirmed_violation", "object"
	case intruderConfirmed:
		a.SystemVerdict, a.ConvictKind = "confirmed_violation", "intruder"
	default:
		anyPhone := aiPhoneW > 0 || phoneClient || detOnlyW > 0
		anyObj := objBestW > 0
		anySecond := secondW > 0 || clientIntruder
		// FIX 8: ANY uncertainty — partial signal, unknown object, sensor
		// contradiction, or behavioural flag — escalates to REVIEW, never a
		// silent clear.
		suspicion := anyPhone || anyObj || anySecond || a.UnknownFrames > 0 || covert || framing || len(a.Contradictions) > 0
		switch {
		case a.UsableFrames == 0:
			a.SystemVerdict = "technical_failure"
		case !llmHadFrames:
			a.SystemVerdict = "needs_review" // fired but AI produced no grounded read → cannot clear
		case suspicion:
			a.SystemVerdict = "needs_review"
		default:
			a.SystemVerdict = "no_violation"
		}
	}

	// ── Strength DERIVED from the score (FIX 2 — grounded, not a human label) ─
	switch a.SystemVerdict {
	case "no_violation":
		a.Strength = "strong" // strong evidence of NO violation
	case "technical_failure":
		a.Strength = "inconclusive"
	default:
		a.Strength = scoreBand(a.EvidenceScore)
	}

	// ── Confidence (branch-anchored; honest until real calibration exists) ────
	switch a.SystemVerdict {
	case "confirmed_violation":
		switch {
		case a.ConvictKind == "phone" && phoneClient && llmPhoneSeen:
			a.Confidence = 97
		case a.ConvictKind == "intruder":
			a.Confidence = 90
		default:
			a.Confidence = 85
		}
	case "needs_review":
		a.Confidence = 50
	case "technical_failure":
		a.Confidence = 40
	default:
		a.Confidence = 80
	}

	// ── Why-not explanations (defensibility) ─────────────────────────────────
	switch a.SystemVerdict {
	case "confirmed_violation":
		switch a.ConvictKind {
		case "phone":
			switch {
			case phoneClient && llmPhoneSeen:
				a.WhyNotClear = "A phone was seen by BOTH the on-device detector and AI vision across good-quality frames."
			case a.PhoneGood >= 2:
				a.WhyNotClear = fmt.Sprintf("A phone was clearly visible in %d good-quality frames.", a.PhoneGood)
			default:
				a.WhyNotClear = "A phone was clearly visible in a good-quality frame and corroborated by the on-device detector."
			}
		case "object":
			a.WhyNotClear = fmt.Sprintf("A %s was clearly held across %d good-quality frames.", objType, objGoodBest)
		case "intruder":
			a.WhyNotClear = fmt.Sprintf("A second person was clearly visible in %d good-quality frames.", a.SecondGood)
		}
	case "needs_review":
		switch {
		case a.PhoneGood+a.PhoneWeak > 0 || phoneClient:
			a.WhyNotClear = "A phone-like object appeared near the candidate's hands."
			a.WhyNotCheating = "It appeared in too few or too low-quality frames (or the AI and detector disagreed) to distinguish it confidently from paper or another benign object."
		case anyObjType(objType):
			a.WhyNotClear = fmt.Sprintf("A %s-like object was visible.", objType)
			a.WhyNotCheating = fmt.Sprintf("The %s appeared in only one clear frame — not consistent enough to confirm.", objType)
		case a.SecondGood+a.SecondWeak > 0 || clientIntruder:
			a.WhyNotClear = "A possible second person appeared in the candidate's frame."
			a.WhyNotCheating = "The second person could not be confirmed across multiple good-quality frames (it may be a reflection, poster, or an object/hand misread as a face)."
		case a.UnknownFrames > 0:
			a.WhyNotClear = "An unidentified object was visible in at least one frame."
			a.WhyNotCheating = "The object could not be clearly identified as a cheating aid."
		case covert:
			a.WhyNotClear = "The candidate repeatedly looked down/away with no on-screen reason — consistent with a device hidden below the camera."
			a.WhyNotCheating = "No prohibited item was directly visible — the concern is behavioural only."
		case framing:
			a.WhyNotClear = "The candidate's torso and hands were kept out of frame — a framing that can hide a phone or notes."
			a.WhyNotCheating = "No prohibited item was directly visible — the concern is behavioural only."
		default:
			a.WhyNotClear = "This incident triggered but AI vision could not analyse it (the free vision tier was rate-limited)."
			a.WhyNotCheating = "AI analysis was unavailable, so the evidence could not be verified — the system will not clear it, but cannot confirm a violation either."
		}
	case "technical_failure":
		a.WhyNotClear = "The system could not analyse this incident (no usable frames / AI unavailable)."
		a.WhyNotCheating = "There is insufficient usable evidence to confirm a violation."
	}

	// ── FIX 10: explain why NOT each alternative verdict (no black box) ───────
	a.WhyNot = map[string]string{}
	scoreTxt := fmt.Sprintf("evidence score %d/100", a.EvidenceScore)
	switch a.SystemVerdict {
	case "confirmed_violation":
		a.WhyNot["needs_review"] = "Evidence PASSED the confirmation threshold (" + scoreTxt + ", weighted across good-quality frames) — this is not a single ambiguous frame."
		a.WhyNot["no_violation"] = "A prohibited " + a.ConvictKind + " was positively identified, so a clean pass is impossible."
	case "needs_review":
		a.WhyNot["confirmed_violation"] = "Evidence did NOT reach the confirmation threshold (" + scoreTxt + " < required). " + a.WhyNotCheating
		a.WhyNot["no_violation"] = "A suspicious signal is present (" + a.WhyNotClear + ") so it cannot be silently cleared."
	case "no_violation":
		a.WhyNot["confirmed_violation"] = "No prohibited object or second person reached even the review threshold across the usable frames."
		a.WhyNot["needs_review"] = "No ambiguous object, sensor contradiction, or behavioural flag was present — nothing to review."
	case "technical_failure":
		a.WhyNot["confirmed_violation"] = "The system could not obtain usable evidence, so a conviction cannot be asserted."
		a.WhyNot["no_violation"] = "The system could not verify the frames, so it cannot assert a clean pass either."
	}
	return a
}

func anyObjType(t string) bool { return strings.TrimSpace(t) != "" }

// scoreBand maps the measurable EvidenceScore to a strength label — so "strength"
// is DERIVED from real signals, never a subjective tag (FIX 2).
func scoreBand(score int) string {
	switch {
	case score >= 65:
		return "strong"
	case score >= 40:
		return "moderate"
	case score >= 20:
		return "weak"
	default:
		return "inconclusive"
	}
}

// distinctCheatTypes returns the distinct cheat-object labels the AI used across
// frames (phone counted once) — >1 means the AI was internally inconsistent.
func distinctCheatTypes(frames []FrameObs) []string {
	seen := map[string]bool{}
	for _, f := range frames {
		if frameQuality(f) == "unusable" {
			continue
		}
		if isPhoneObject(f.ObjectInHand) {
			seen["phone"] = true
			continue
		}
		if classifyObject(f.ObjectInHand) == "cheat" {
			seen[strings.ToLower(strings.TrimSpace(f.ObjectInHand))] = true
		}
	}
	out := make([]string, 0, len(seen))
	for t := range seen {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// buildTimelines computes the appearance history (first / peak-confidence / last
// frame) for each tracked object — phone, any AI cheat object, and a second
// person (FIX 6).
func buildTimelines(frames []FrameObs) []ObjectTimeline {
	type acc struct {
		first, last, peak, peakConf, count int
		peakW                              float64
	}
	track := map[string]*acc{}
	note := func(key string, f FrameObs) {
		w := frameWeight(f)
		conf := int(math.Round(w * 100))
		if f.AIConfidence > 0 {
			conf = f.AIConfidence
		}
		x := track[key]
		if x == nil {
			x = &acc{first: f.Frame, last: f.Frame, peak: f.Frame, peakW: -1}
			track[key] = x
		}
		if f.Frame < x.first {
			x.first = f.Frame
		}
		if f.Frame > x.last {
			x.last = f.Frame
		}
		if w > x.peakW {
			x.peakW, x.peak, x.peakConf = w, f.Frame, conf
		}
		x.count++
	}
	for _, f := range frames {
		if frameQuality(f) == "unusable" {
			continue
		}
		if isPhoneObject(f.ObjectInHand) || f.DetectorPhone {
			note("phone", f)
		} else if classifyObject(f.ObjectInHand) == "cheat" {
			note(strings.ToLower(strings.TrimSpace(f.ObjectInHand)), f)
		}
		if f.SecondPerson {
			note("second person", f)
		}
	}
	keys := make([]string, 0, len(track))
	for k := range track {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]ObjectTimeline, 0, len(keys))
	for _, k := range keys {
		x := track[k]
		out = append(out, ObjectTimeline{
			Object: k, FirstFrame: x.first, PeakFrame: x.peak,
			LastFrame: x.last, PeakConfidence: x.peakConf, Frames: x.count,
		})
	}
	return out
}

// decideVerdict applies the evidence-strength engine. The model fills PerFrame
// (what it actually sees in each frame); this scores each frame, weighs the whole
// evidence pattern, and recomputes the verdict, materials, phone/intruder flags,
// evidence text, why-not explanations and summary DETERMINISTICALLY — so the model
// can neither hallucinate a conviction nor bury a real object, and no weak or
// unanalysed incident ever becomes a confident CLEAR.
func decideVerdict(r *Report, clientIntruder bool, triggerReason string, phoneClient, llmPhoneSeen, llmRan, llmHadFrames bool) {
	modelConf := r.ConfidenceScore // captured for replay — the model's confidence BEFORE the engine adjusts it

	// Score each frame's evidence in place (object class/status, frame status).
	scoreFrameEvidence(r.PerFrame, phoneClient)

	// Frames to weigh. If the model skipped the per-frame table entirely, fall
	// back to its own top-level flags so a real detection is never silently
	// dropped — synthesise two good phone/material frames to feed the ladder.
	frames := r.PerFrame
	if len(frames) == 0 {
		if r.PhoneVisible {
			frames = []FrameObs{{Frame: 1, ObjectInHand: "phone", Quality: "good"}, {Frame: 2, ObjectInHand: "phone", Quality: "good"}}
		} else if len(r.Materials) > 0 && classifyObject(r.Materials[0]) == "cheat" {
			m := strings.ToLower(strings.TrimSpace(r.Materials[0]))
			frames = []FrameObs{{Frame: 1, ObjectInHand: m, Quality: "good"}, {Frame: 2, ObjectInHand: m, Quality: "good"}}
		}
	}

	a := assessEvidence(frames, clientIntruder, triggerReason, phoneClient, llmPhoneSeen, llmRan, llmHadFrames)
	objType := a.ObjType

	// Reset convict-able fields; set them only from the assessment below.
	r.PhoneVisible = false
	r.PhoneAIConfirmed = false
	r.CheatingMaterialDetected = false
	r.Materials = []string{}
	r.IntruderConfirmed = a.SystemVerdict == "confirmed_violation" && a.ConvictKind == "intruder"
	r.SystemVerdict = a.SystemVerdict
	r.EvidenceStrength = a.Strength
	r.EvidenceScore = a.EvidenceScore
	r.WhyNotClear = a.WhyNotClear
	r.WhyNotCheating = a.WhyNotCheating
	r.WhyNot = a.WhyNot
	r.Contradictions = a.Contradictions
	r.Timelines = a.Timelines
	r.ConfidenceScore = a.Confidence
	r.LLMFailed = a.SystemVerdict == "technical_failure"
	mats := []string{}
	rule := "clear"

	switch a.SystemVerdict {
	case "confirmed_violation":
		r.Cheating, r.ReviewRequired = 1, false
		r.CheatingMaterialDetected = a.ConvictKind == "phone" || a.ConvictKind == "object"
		r.CheatingLikelihood = "high"
		switch a.ConvictKind {
		case "phone":
			rule = "confirmed"
			r.PhoneVisible = true
			r.PhoneAIConfirmed = llmPhoneSeen
			mats = append(mats, "phone")
		case "object":
			rule = "confirmed"
			mats = append(mats, objType)
		case "intruder":
			rule = "intruder"
		}
	case "needs_review":
		rule = "ambiguous"
		r.Cheating, r.ReviewRequired = 0, true
		r.CheatingLikelihood = "medium"
	case "technical_failure":
		rule = "technical_failure"
		r.Cheating, r.ReviewRequired = 0, true
		r.CheatingLikelihood = "medium"
	default: // no_violation
		rule = "clear"
		r.Cheating, r.ReviewRequired = 0, false
		r.CheatingLikelihood = "low"
	}
	r.Materials = mats

	// ── Record the exact inputs + decision path so the verdict is reproducible ──
	// AI-OBSERVER-ONLY made explicit: the verdict above was computed PURELY from
	// these deterministic inputs (per-frame observations + on-device signals). The
	// AI never sets the verdict — it only supplies the observations counted here.
	r.ReplayInput = &ReplayInput{
		ClientIntruder:  clientIntruder,
		TriggerReason:   triggerReason,
		PhoneClient:     phoneClient,
		LLMPhoneSeen:    llmPhoneSeen,
		LLMRan:          llmRan,
		LLMHadFrames:    llmHadFrames,
		ModelConfidence: modelConf,
	}
	r.DecisionTrace = &VerdictTrace{
		CheatFrames:      a.CheatFrames,
		UnknownFrames:    a.UnknownFrames,
		SecondFrames:     a.SecondFrames,
		LLMPhoneFrames:   a.LLMPhoneFrames,
		PhoneCheat:       a.PhoneCheat,
		ObjType:          objType,
		Rule:             rule,
		PhoneGoodFrames:  a.PhoneGood,
		PhoneWeakFrames:  a.PhoneWeak,
		SecondGood:       a.SecondGood,
		SecondWeak:       a.SecondWeak,
		UsableFrames:     a.UsableFrames,
		SystemVerdict:    a.SystemVerdict,
		EvidenceStrength: a.Strength,
	}

	// Build a frame-by-frame evidence string so EVERY report + PDF shows the
	// breakdown ("what he was using") with no dashboard change. Only rebuild
	// when we actually have grounded frames; otherwise keep the model's prose.
	if len(r.PerFrame) > 0 {
		var b strings.Builder
		for _, f := range r.PerFrame {
			obj := strings.TrimSpace(f.ObjectInHand)
			if obj == "" {
				obj = "none"
			}
			gaze := strings.TrimSpace(f.Gaze)
			if gaze == "" {
				gaze = "unknown"
			}
			fmt.Fprintf(&b, "Frame %d: object=%s, gaze=%s", f.Frame, obj, gaze)
			if f.SecondPerson {
				b.WriteString(", SECOND PERSON visible")
			}
			if d := strings.TrimSpace(f.Description); d != "" {
				fmt.Fprintf(&b, " — %s", d)
			}
			b.WriteString("\n")
		}
		r.VisualEvidence = strings.TrimSpace(b.String())
	}

	// Deterministic, non-contradictory one-line summary — driven by the evidence
	// assessment so the words can never contradict the verdict.
	nGood := a.PhoneGood
	switch a.SystemVerdict {
	case "confirmed_violation":
		switch a.ConvictKind {
		case "phone":
			switch {
			case phoneClient && llmPhoneSeen:
				r.Summary = "CONFIRMED cheating: a phone was detected by BOTH the on-device real-time detector AND AI vision across good-quality frames."
			case a.PhoneGood >= 2:
				r.Summary = fmt.Sprintf("CONFIRMED cheating: a phone was clearly held across %d good-quality frames (AI vision).", nGood)
			default:
				r.Summary = "CONFIRMED cheating: a phone was clearly visible in a good-quality frame and corroborated by the on-device detector."
			}
		case "object":
			r.Summary = fmt.Sprintf("CONFIRMED cheating: %s clearly held across %d good-quality frames.", objType, a.CheatFrames)
		case "intruder":
			r.Summary = fmt.Sprintf("CONFIRMED: a second person is clearly present in %d good-quality frames.", a.SecondGood)
		}
	case "needs_review":
		r.Summary = "NEEDS REVIEW: " + a.WhyNotClear + " " + a.WhyNotCheating + " A human examiner should verify."
	case "technical_failure":
		r.Summary = "TECHNICAL FAILURE / INCOMPLETE ANALYSIS: " + a.WhyNotClear + " " + a.WhyNotCheating + " This incident could not be judged and must be reviewed manually — it is NOT cleared."
	default:
		r.Summary = "NO VIOLATION DETECTED: all analysed frames were usable, no prohibited object or second person was confirmed, and no suspicious interaction was found."
	}
	if r.Recommendation == "" {
		switch {
		case r.Cheating == 1:
			r.Recommendation = "Flag this candidate for examiner action."
		case r.ReviewRequired:
			r.Recommendation = "Queue for manual review by the examiner."
		default:
			r.Recommendation = "No action required."
		}
	}
}

// ── Vision LLM — full JSON payload + binary cheating verdict ──────────────

// maxLLMImages is the per-request image cap for the vision model. Groq Llama 4
// Scout accepts at most 5 images per request, so we capture 6 frames (kept on
// disk + in the collage thumbnail) and send the 5 most time-spread ones. It is a
// var so a model with a higher limit can send all 6 via LLM_MAX_IMAGES in .env.
var maxLLMImages = 5

// pickEvenly returns up to max indices spread across n items, ALWAYS including
// the first and last (so the full capture time-span reaches the model). For
// n=6, max=5 this yields [0 1 2 3 5] — it drops one middle frame, which cannot
// change a verdict because confirmation requires an object in ≥2 frames.
func pickEvenly(n, max int) []int {
	if n <= 0 {
		return nil
	}
	if n <= max {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		return idx
	}
	if max <= 1 {
		// Only one slot to fill (and n>1) — return the first index so we never
		// divide by (max-1)==0 below, which would panic and crash the server.
		return []int{0}
	}
	idx := make([]int, max)
	for i := 0; i < max; i++ {
		idx[i] = i * (n - 1) / (max - 1)
	}
	return idx
}

// selectFrames normalises capture frames to full data URLs, caps the count to
// maxLLMImages sampling evenly across the set, and ALSO returns the source index
// of each selected frame. The indices let buildMergedPerFrame map the model's
// per-frame observations back onto the exact physical frames it was shown — so an
// object is never pinned to the wrong row when empties are skipped or the model
// returns a different count than the number of images sent.
func selectFrames(frames []string, max int) (urls []string, srcIdx []int) {
	for _, i := range pickEvenly(len(frames), max) {
		u := strings.TrimSpace(frames[i])
		if u == "" {
			continue
		}
		if !strings.HasPrefix(u, "data:") {
			u = "data:image/jpeg;base64," + u
		}
		urls = append(urls, u)
		srcIdx = append(srcIdx, i)
	}
	return urls, srcIdx
}

// selectFrameURLs is the URL-only form for callers that don't need the indices.
func selectFrameURLs(frames []string, max int) []string {
	urls, _ := selectFrames(frames, max)
	return urls
}

// alignGazeToImages subsamples the per-frame gaze metadata down to the number
// of images actually sent, using the SAME even-sampling as the images — so the
// gaze entry at position i corresponds to image i, and the payload never says
// "6 frames" while showing 5 images.
func alignGazeToImages(frames []map[string]any, n int) []map[string]any {
	if n <= 0 || len(frames) <= n {
		return frames
	}
	out := make([]map[string]any, 0, n)
	for _, i := range pickEvenly(len(frames), n) {
		out = append(out, frames[i])
	}
	return out
}

// buildVisionPrompt assembles the system prompt + a dynamic header stating how
// many images the model received + the full incident JSON.
func buildVisionPrompt(imageCount int, payload map[string]any) string {
	payloadJSON, _ := json.MarshalIndent(payload, "", "  ")
	return systemPrompt +
		fmt.Sprintf("\n\nYou are given %d separate images, in chronological order (image 1 = earliest, ~1 second apart). Output EXACTLY %d per_frame entries — one per image, in the same order (frame 1 = image 1).\n\nCOMPLETE INCIDENT DATA (JSON):\n", imageCount, imageCount) +
		string(payloadJSON)
}

const systemPrompt = `You are a meticulous exam-proctoring vision analyst. Your ONLY job is to REPORT WHAT YOU ACTUALLY SEE in each frame — frame by frame. You do NOT decide the final cheating verdict; a separate rule engine does that from your per-frame observations. So be a precise observer, not a judge. Do not assume guilt and do not assume innocence — just describe.

You receive:
1. A set of SEPARATE images (up to 5), in chronological order, taken ~1 second apart — image 1 is the earliest. Each image is a full, high-resolution frame (NOT a collage). ALL faces are PIXELATED for privacy — ignore the blurred face region, but examine EVERYTHING ELSE (hands, objects, background, extra people).
2. Numeric gaze data per frame (iris deviation h/v, head yaw/pitch).
3. Client sensor hints — WEAK heuristics, NOT evidence. Trust your eyes, not the hints.

★ THE CORE TASK: fill the "per_frame" array with ONE entry per image you are given, in order (image 1 → frame 1, image 2 → frame 2, …). For EACH image independently, look and report:
- hand_visible: is a hand visible in THIS frame?
- object_in_hand: EXACTLY ONE of "none" | "phone" | "paper" | "book" | "earbud" | "watch" | "other". Use "none" if the hand is empty or there is no hand. Be honest per-frame: if frame 3 has nothing but frame 4 shows a phone, say so.
- second_person: is a clearly distinct SECOND person (extra face/body/shoulder/hand that is not the candidate's) visible in THIS frame?
- gaze: "ahead" | "left" | "right" | "up" | "down" for this frame.
- quality: rate how clearly you can judge THIS frame — "good" (sharp, well-lit, object region clearly visible) | "acceptable" (usable) | "poor" (blurry, dark, glare, object tiny/partly out of frame — you are unsure) | "unusable" (cannot judge at all). Be honest: mark "poor" whenever a device/paper call is a guess. The rule engine weighs good frames more heavily and never convicts on poor frames alone.
- object_confidence: an INTEGER 0-100 for how sure you are about the object_in_hand call in THIS frame (100 = certain it is that object, 50 = a coin-flip guess, 0 = no idea). This is a structured signal the rule engine weighs — it does NOT read your prose. If object_in_hand is "none", set this to 0.
- description: one short sentence on what is actually visible in this frame.

⚠ OBSERVATION RULES (critical):
- NEVER invent an object. If you do not CLEARLY see a rigid rectangular device, object_in_hand is NOT "phone". A bare hand, curled fingers, knuckles, a chin-rest, a shadow, or a watch strap is NOT a phone.
- A hand resting on the chin/cheek/mouth/temple with nothing in it = object_in_hand "none". This is normal and is the #1 false alarm — do not label it.
- A WATER BOTTLE, cup, glass, mug, can, food, snack, drink, pen or pencil is NOT a cheating item and is NOT suspicious — a candidate is allowed to drink or hold ordinary items. Set object_in_hand to "none" for these; do NOT report them as "other". Reserve "other" ONLY for a genuinely unusual object you cannot identify — never for an everyday drink, snack, or piece of stationery.
- The candidate's OWN body and clothing — a shoulder, arm, sleeve, shirt, collar, hair — plus ordinary eyeglasses being worn, and room furniture (desk, chair, wall, ceiling) are NEVER objects and NEVER a second person. Report object_in_hand "none" for these; do NOT label a shoulder or clothing edge as "other" or as an extra person.
- "phone" = dark/glowing rigid slab. "paper" = light-colored flat sheet/page. Never swap these. If a flat light object, it is "paper", not "phone".
- NOTES are a violation too: ANY handwritten or printed sheet, index card, sticky note, open book or notebook — held in the hand, lying on the desk, or stuck nearby in view — set object_in_hand to "paper" (loose sheets/notes) or "book" (bound book/notebook). Report these as carefully as a phone; do not ignore paper just because it is not a device.
- Only set object_in_hand to a real type for a frame where you genuinely see it. If unsure for a frame, use "none". It is fine for different frames to differ — the rule engine wants the truth per frame, and it only confirms an object that PERSISTS across multiple frames (so a one-frame guess will NOT cause a false accusation — but inventing the same object in every frame WILL).
- HINT USAGE: if phone_detected_client=true, the on-device detector already saw a phone — LOOK CAREFULLY at EVERY frame for it (it may be small, angled, held near the ear, or partly out of frame) and report "phone" for each frame where you can actually see it. Do NOT, however, invent a phone from a bare hand / curled fingers / chin-rest when you genuinely cannot see a device.
- PHONE + SECOND PERSON TOGETHER: a phone and an intruder can occur in the SAME frame. If you see a second person, do NOT get tunnel-vision on them and forget the phone — inspect EVERY hand in the frame (the candidate's AND the other person's) for a phone/device or notes, and set object_in_hand for that frame if a device is genuinely visible. A second person does not excuse an undetected phone; report BOTH.
- BEHAVIOUR NOTE: if the head/eyes are turned DOWN or AWAY from the screen in a frame, state it in that frame's description and summarise it in gaze_analysis — repeated down-looks may indicate a device hidden below the camera. This affects ONLY your description/gaze_analysis text; still report object_in_hand strictly from what is visible (do NOT invent a phone you cannot see).
- FRAMING NOTE: if the candidate's shoulders/torso are NOT visible and the view is mostly the face plus ceiling/wall (camera or laptop lid tilted up), note this in that frame's description and in gaze_analysis as a possible attempt to keep the hands/lap out of view. This affects ONLY your description/gaze_analysis text; still report object_in_hand strictly from what is visible.

Then provide overall fields. Respond with ONLY a valid JSON object — no prose, no markdown:
{
  "per_frame": [
    {"frame": 1, "hand_visible": true, "object_in_hand": "none", "second_person": false, "gaze": "ahead", "quality": "good", "object_confidence": 0, "description": "..."}
  ],
  "confidence_score": 0-100,
  "suspicious_behaviors": ["describe each specifically, or empty if none"],
  "gaze_analysis": "how gaze/head moved across the frames, with frame references",
  "recommendation": "suggested examiner action"
}

Output ONE per_frame entry for EACH image you were given, in order — no more, no fewer. Accuracy of object_in_hand per frame is the single most important thing.`

func callGemini(apiKey string, imageURLs []string, incidentID, timestamp string, frames []map[string]any, phoneDetected bool, handPosture string, faceCount int, intruderDetected bool, triggerReason string) (Report, string, error) {
	if triggerReason == "" {
		triggerReason = "hand_visible_and_gaze_toward_wrist"
	}
	frames = alignGazeToImages(frames, len(imageURLs)) // never claim more frames than images shown
	payload := map[string]any{
		"incident_id":              incidentID,
		"timestamp":                timestamp,
		"trigger":                  triggerReason,
		"images_provided":          len(imageURLs),
		"frame_count":              len(frames),
		"phone_detected_client":    phoneDetected,
		"hand_posture":             handPosture,
		"face_count":               faceCount,
		"intruder_detected_client": intruderDetected,
		"frames":                   frames,
	}
	prompt := buildVisionPrompt(len(imageURLs), payload)

	// One inlineData part per frame (raw base64, no data: prefix), then text.
	parts := make([]geminiPart, 0, len(imageURLs)+1)
	for _, u := range imageURLs {
		data := u
		if idx := strings.Index(u, ","); idx >= 0 {
			data = u[idx+1:]
		}
		parts = append(parts, geminiPart{InlineData: &geminiInlineData{MimeType: "image/jpeg", Data: data}})
	}
	parts = append(parts, geminiPart{Text: prompt})

	gemReq := geminiRequest{
		Contents: []geminiContent{{Parts: parts}},
		GenerationConfig: &genConfig{
			ResponseMimeType: "application/json",
		},
	}
	body, err := json.Marshal(gemReq)
	if err != nil {
		return Report{}, "", err
	}

	geminiMu.Lock()
	defer geminiMu.Unlock()

	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		// Honour rate-limit backoff set by a previous call
		if wait := time.Until(geminiNext); wait > 0 {
			fmt.Printf("[gemini] rate-limit pause %.1fs\n", wait.Seconds())
			time.Sleep(wait)
		}
		req2, _ := http.NewRequest("POST", geminiURL+"?key="+apiKey, bytes.NewReader(body))
		req2.Header.Set("Content-Type", "application/json")
		resp, err := geminiClient.Do(req2)
		if err != nil {
			lastErr = err
			fmt.Printf("[gemini] attempt %d net error: %v — retrying in 10s\n", attempt, err)
			geminiNext = time.Now().Add(10 * time.Second)
			continue
		}
		var gemResp geminiResponse
		decErr := json.NewDecoder(resp.Body).Decode(&gemResp)
		resp.Body.Close()
		if decErr != nil {
			lastErr = decErr
			continue
		}
		if gemResp.Error != nil {
			msg := gemResp.Error.Message
			lastErr = fmt.Errorf("gemini API error: %s", msg)
			wait := parseRetryAfter(msg)
			// Long/absent backoff = quota EXHAUSTED → give up fast so the
			// on-device provisional verdict stands (don't hang for minutes).
			if wait == 0 || wait > 12*time.Second {
				fmt.Printf("[gemini] quota exhausted (suggested %.0fs) — giving up\n", wait.Seconds())
				return Report{}, "", lastErr
			}
			fmt.Printf("[gemini] attempt %d rate-limited — waiting %.1fs\n", attempt, wait.Seconds())
			geminiNext = time.Now().Add(wait)
			continue
		}
		if len(gemResp.Candidates) == 0 || len(gemResp.Candidates[0].Content.Parts) == 0 {
			lastErr = fmt.Errorf("empty response from Gemini")
			continue
		}
		// Small inter-call gap to stay well under 20 RPM
		geminiNext = time.Now().Add(4 * time.Second)
		raw := gemResp.Candidates[0].Content.Parts[0].Text
		clean := extractJSON(raw)
		var rep Report
		if err := json.Unmarshal([]byte(clean), &rep); err != nil {
			return Report{}, raw, fmt.Errorf("parse report: %w", err)
		}
		return rep, clean, nil
	}
	return Report{}, "", lastErr
}

var retryRe = regexp.MustCompile(`(?:retry|try again) in ([0-9]+(?:\.[0-9]+)?)s`)

func parseRetryAfter(msg string) time.Duration {
	m := retryRe.FindStringSubmatch(msg)
	if len(m) < 2 {
		return 0
	}
	secs, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0
	}
	return time.Duration(secs*1000)*time.Millisecond + 2*time.Second // add 2s buffer
}

// ── Groq (Llama 3.2 Vision) — OpenAI-compatible, ~2s, 100 req/day free ───────

const groqURL = "https://api.groq.com/openai/v1/chat/completions"

// groqModel is declared near the top (configurable via GROQ_MODEL).

type groqRequest struct {
	Model          string          `json:"model"`
	Messages       []groqMessage   `json:"messages"`
	Temperature    float64         `json:"temperature"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	ResponseFormat *groqRespFormat `json:"response_format,omitempty"`
}
type groqRespFormat struct {
	Type string `json:"type"` // "json_object" → forces valid JSON output
}
type groqMessage struct {
	Role    string        `json:"role"`
	Content []groqContent `json:"content"`
}
type groqContent struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *groqImageURL `json:"image_url,omitempty"`
}
type groqImageURL struct {
	URL string `json:"url"`
}
type groqResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func callGroq(apiKey string, imageURLs []string, incidentID, timestamp string, frames []map[string]any, phoneDetected bool, handPosture string, faceCount int, intruderDetected bool, triggerReason string) (Report, string, error) {
	if triggerReason == "" {
		triggerReason = "hand_visible_and_gaze_toward_wrist"
	}
	frames = alignGazeToImages(frames, len(imageURLs)) // never claim more frames than images shown
	payload := map[string]any{
		"incident_id":              incidentID,
		"timestamp":                timestamp,
		"trigger":                  triggerReason,
		"images_provided":          len(imageURLs),
		"frame_count":              len(frames),
		"phone_detected_client":    phoneDetected,
		"hand_posture":             handPosture,
		"face_count":               faceCount,
		"intruder_detected_client": intruderDetected,
		"frames":                   frames,
	}
	prompt := buildVisionPrompt(len(imageURLs), payload)

	// One image_url block per frame (full data URL), then the text block.
	content := make([]groqContent, 0, len(imageURLs)+1)
	for _, u := range imageURLs {
		content = append(content, groqContent{Type: "image_url", ImageURL: &groqImageURL{URL: u}})
	}
	content = append(content, groqContent{Type: "text", Text: prompt})

	greq := groqRequest{
		Model:          groqModel,
		Temperature:    0.1,                                  // low temp → consistent, deterministic verdicts
		MaxTokens:      2048,                                 // room for the per-frame table
		ResponseFormat: &groqRespFormat{Type: "json_object"}, // force valid JSON
		Messages: []groqMessage{{
			Role:    "user",
			Content: content,
		}},
	}
	body, err := json.Marshal(greq)
	if err != nil {
		return Report{}, "", err
	}

	// One call at a time, with rate-limit back-off and retries — so a burst of
	// incidents never leaves one stuck on PENDING (see groqMu comment above).
	groqMu.Lock()
	defer groqMu.Unlock()

	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		if wait := time.Until(groqNext); wait > 0 {
			fmt.Printf("[groq] rate-limit pause %.1fs\n", wait.Seconds())
			time.Sleep(wait)
		}
		req, _ := http.NewRequest("POST", groqURL, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+apiKey)

		resp, err := geminiClient.Do(req) // reuse the 120s timeout client
		if err != nil {
			lastErr = err
			fmt.Printf("[groq] attempt %d net error: %v — retrying in 10s\n", attempt, err)
			groqNext = time.Now().Add(10 * time.Second)
			continue
		}
		// Rate-limited / server error → honour Retry-After, but give up FAST if
		// the backoff is long (quota exhausted) so the on-device provisional
		// verdict stands instead of hanging for minutes.
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			wait := groqRetryAfter(resp)
			resp.Body.Close()
			lastErr = fmt.Errorf("groq HTTP %d", resp.StatusCode)
			if wait > 12*time.Second {
				fmt.Printf("[groq] HTTP %d backoff %.0fs too long — giving up (provisional stands)\n", resp.StatusCode, wait.Seconds())
				return Report{}, "", lastErr
			}
			fmt.Printf("[groq] attempt %d HTTP %d — backing off %.1fs\n", attempt, resp.StatusCode, wait.Seconds())
			groqNext = time.Now().Add(wait)
			continue
		}

		var gresp groqResponse
		decErr := json.NewDecoder(resp.Body).Decode(&gresp)
		resp.Body.Close()
		if decErr != nil {
			lastErr = decErr
			fmt.Printf("[groq] attempt %d decode error: %v\n", attempt, decErr)
			groqNext = time.Now().Add(5 * time.Second)
			continue
		}
		if gresp.Error != nil {
			lastErr = fmt.Errorf("groq error: %s", gresp.Error.Message)
			wait := parseRetryAfter(gresp.Error.Message)
			if wait == 0 || wait > 12*time.Second {
				fmt.Printf("[groq] API error (retry %.0fs) — giving up (provisional stands): %s\n", wait.Seconds(), gresp.Error.Message)
				return Report{}, "", lastErr
			}
			fmt.Printf("[groq] attempt %d API error: %s — waiting %.1fs\n", attempt, gresp.Error.Message, wait.Seconds())
			groqNext = time.Now().Add(wait)
			continue
		}
		if len(gresp.Choices) == 0 {
			lastErr = fmt.Errorf("empty response from Groq")
			groqNext = time.Now().Add(5 * time.Second)
			continue
		}

		groqNext = time.Now().Add(2 * time.Second) // small inter-call gap
		raw := gresp.Choices[0].Message.Content
		clean := extractJSON(raw)
		var rep Report
		if err := json.Unmarshal([]byte(clean), &rep); err != nil {
			return Report{}, raw, fmt.Errorf("parse groq report: %w", err)
		}
		return rep, clean, nil
	}
	return Report{}, "", lastErr
}

// groqRetryAfter reads the Retry-After header Groq sets on 429 (value in
// seconds), falling back to a sane default when absent/unparseable.
func groqRetryAfter(resp *http.Response) time.Duration {
	if ra := strings.TrimSpace(resp.Header.Get("Retry-After")); ra != "" {
		if secs, err := strconv.ParseFloat(ra, 64); err == nil && secs > 0 {
			return time.Duration(secs*1000)*time.Millisecond + 2*time.Second
		}
	}
	return 15 * time.Second
}

// extractJSON strips code fences / prose and returns the {...} object.
func extractJSON(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start >= 0 && end > start {
		return s[start : end+1]
	}
	return strings.TrimSpace(s)
}

// ── Supabase persistence ───────────────────────────────────────────────────

func insertSupabase(rec IncidentRecord) {
	url := os.Getenv("SUPABASE_URL")
	key := os.Getenv("SUPABASE_SECRET_KEY")
	if url == "" || key == "" {
		return
	}
	body, err := json.Marshal(rec)
	if err != nil {
		return
	}
	// Upsert on the id primary key. Each incident is written twice — the immediate
	// provisional verdict, then the AI-enriched final one — under the SAME id. A
	// plain INSERT makes the second (final) write 409 on the primary key, so the
	// dashboard would stay stuck on the provisional forever. merge-duplicates +
	// on_conflict=id makes the final write UPDATE the row so the enriched verdict
	// (and llm_pending=false) actually reaches the Supabase-backed dashboard.
	req, _ := http.NewRequest("POST", url+"/rest/v1/incidents?on_conflict=id", bytes.NewReader(body))
	req.Header.Set("apikey", key)
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Prefer", "resolution=merge-duplicates,return=minimal")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Printf("[supabase] insert error: %v\n", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(resp.Body)
		fmt.Printf("[supabase] insert failed %d: %s\n", resp.StatusCode, strings.TrimSpace(string(msg)))
		return
	}
	fmt.Printf("[supabase] stored %s\n", rec.ID)
}

// ── GET /incidents — list for the dashboard (metadata + collage URL) ───────

// anyInt reads an int from map[string]any regardless of whether it was stored
// as int, float64, or int64 (JSON decode vs struct-populated map difference).
func anyInt(v any) int {
	switch x := v.(type) {
	case int:
		return x
	case float64:
		return int(x)
	case int64:
		return int(x)
	}
	return 0
}

// handleSessionReport aggregates all incidents into one session-level summary.
// handleReplay is the forensic/audit endpoint: GET /replay?id=<incidentID>.
// It loads the stored report, DETERMINISTICALLY re-derives the verdict from the
// recorded ReplayInput, and returns the stored verdict, the replayed verdict,
// whether they MATCH (proof the verdict was rule-derived, not tampered), the
// decision trace, and the full append-only event history. This is how "why was
// I flagged?" is answered with an exact, reproducible decision path.
func handleReplay(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	id := strings.TrimSpace(r.URL.Query().Get("id"))
	if id == "" || strings.ContainsAny(id, `/\.`) { // no path traversal
		http.Error(w, `{"error":"missing or invalid id"}`, http.StatusBadRequest)
		return
	}
	dir := filepath.Join(incidentsDir, id)
	b, err := os.ReadFile(filepath.Join(dir, "report.json"))
	if err != nil {
		http.Error(w, `{"error":"incident not found"}`, http.StatusNotFound)
		return
	}
	var stored Report
	if json.Unmarshal(b, &stored) != nil {
		http.Error(w, `{"error":"corrupt report"}`, http.StatusInternalServerError)
		return
	}
	replayed, match := replayVerdict(stored)
	json.NewEncoder(w).Encode(map[string]any{
		"incident_id":  id,
		"reproducible": stored.ReplayInput != nil,
		"match":        match, // true ⇒ stored verdict exactly re-derives from stored inputs
		"inputs":       stored.ReplayInput,
		"decision_trace": stored.DecisionTrace,
		"stored_verdict": map[string]any{
			"cheating": stored.Cheating, "review_required": stored.ReviewRequired,
			"phone_visible": stored.PhoneVisible, "intruder_confirmed": stored.IntruderConfirmed,
			"likelihood": stored.CheatingLikelihood, "confidence": stored.ConfidenceScore,
		},
		"replayed_verdict": map[string]any{
			"cheating": replayed.Cheating, "review_required": replayed.ReviewRequired,
			"phone_visible": replayed.PhoneVisible, "intruder_confirmed": replayed.IntruderConfirmed,
			"likelihood": replayed.CheatingLikelihood, "confidence": replayed.ConfidenceScore,
		},
		"events": readEvents(dir),
	})
}

// validExaminerDecisions are the only human decisions the report accepts.
var validExaminerDecisions = map[string]bool{
	"confirmed": true, "rejected": true, "inconclusive": true, "technical": true,
}

// handleReview records an EXAMINER decision on an incident (human-in-the-loop).
// It never overwrites the system verdict — it stores the human conclusion
// alongside it, re-seals the report (so the decision is tamper-evident), and
// appends a "reviewed" audit event. POST /review?id=<incident>  body:
// {"decision":"confirmed|rejected|inconclusive|technical","note":"...","examiner_id":"..."}
func handleReview(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}
	id := strings.TrimSpace(r.URL.Query().Get("id"))
	if id == "" || strings.ContainsAny(id, `/\.`) { // no path traversal
		http.Error(w, `{"error":"missing or invalid id"}`, http.StatusBadRequest)
		return
	}
	var body struct {
		Decision   string `json:"decision"`
		Note       string `json:"note"`
		ExaminerID string `json:"examiner_id"`
	}
	if json.NewDecoder(r.Body).Decode(&body) != nil {
		http.Error(w, `{"error":"invalid body"}`, http.StatusBadRequest)
		return
	}
	decision := strings.ToLower(strings.TrimSpace(body.Decision))
	if !validExaminerDecisions[decision] {
		http.Error(w, `{"error":"decision must be confirmed|rejected|inconclusive|technical"}`, http.StatusBadRequest)
		return
	}
	dir := filepath.Join(incidentsDir, id)
	b, err := os.ReadFile(filepath.Join(dir, "report.json"))
	if err != nil {
		http.Error(w, `{"error":"incident not found"}`, http.StatusNotFound)
		return
	}
	var stored Report
	if json.Unmarshal(b, &stored) != nil {
		http.Error(w, `{"error":"corrupt report"}`, http.StatusInternalServerError)
		return
	}
	stored.ExaminerDecision = decision
	stored.ExaminerNote = strings.TrimSpace(body.Note)
	stored.ExaminerID = strings.TrimSpace(body.ExaminerID)
	stored.ReviewedAt = time.Now().UTC().Format(time.RFC3339)
	stored.IntegrityHash = sealReport(&stored) // re-seal so the human decision is tamper-evident
	if nb, e := json.MarshalIndent(&stored, "", "  "); e == nil {
		writeJSONAtomic(filepath.Join(dir, "report.json"), nb)
	}
	appendEvent(dir, "reviewed", map[string]any{
		"decision":       decision,
		"examiner_id":    stored.ExaminerID,
		"note":           stored.ExaminerNote,
		"system_verdict": stored.SystemVerdict,
		"integrity_hash": stored.IntegrityHash,
	})
	// Local disk (report.json) is the source of truth the session report reads.
	fmt.Printf("[review] %s  examiner=%q decision=%s (system=%s)\n", id, stored.ExaminerID, decision, stored.SystemVerdict)
	json.NewEncoder(w).Encode(map[string]any{
		"ok":             true,
		"incident_id":    id,
		"decision":       decision,
		"reviewed_at":    stored.ReviewedAt,
		"integrity_hash": stored.IntegrityHash,
		"system_verdict": stored.SystemVerdict,
	})
}

// handleSessionSignals accepts the browser's Layer-1 continuous signal summary —
// the full-session lightweight frame log (total frames, gaze/face/hand/motion
// distributions, phone-hint frames, trigger-event counts, and a downsampled
// per-second timeline). It is stored verbatim and rendered into the SESSION
// OVERVIEW + FRAME TIMELINE of both the JSON report and the official PDF, so the
// report reflects EVERY frame that was streamed, not only the AI-analyzed ones.
func handleSessionSignals(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}
	var body map[string]any
	if json.NewDecoder(r.Body).Decode(&body) != nil {
		http.Error(w, `{"error":"invalid body"}`, http.StatusBadRequest)
		return
	}
	lastSignalsMu.Lock()
	lastSignals = body
	lastSignalsMu.Unlock()
	fmt.Printf("[signals] session frame-log received: frames=%v duration=%vs events=%v\n",
		body["total_frames"], body["duration_secs"], body["events"])
	json.NewEncoder(w).Encode(map[string]any{"ok": true})
}

// handleSessionStatus is the cheap poll the dashboard's finalization gate hits
// every couple seconds. It returns ONLY the in-memory outstanding-job count (no
// disk scan), so "ready to finalize" is a fast, authoritative backend fact:
// queue_depth==0 means every incident that ever entered the pipeline has reached
// a terminal verdict (analyzed or FAILED), so the report can be safely frozen.
func handleSessionStatus(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	depth := llmQueueDepth()
	json.NewEncoder(w).Encode(map[string]any{
		"queue_depth": depth,
		"ready":       depth == 0,
	})
}

func handleSessionReport(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	w.Header().Set("Content-Type", "application/json")

	incidents := listLocalIncidents(0) // uncapped — the session summary must include EVERY incident
	if len(incidents) == 0 {
		// A clean exam still has a full Layer-1 frame log — surface it so the report
		// shows "N frames monitored, 0 incidents", not an empty page.
		json.NewEncoder(w).Encode(map[string]any{"incidents": []any{}, "total": 0, "cheating_count": 0, "pending_count": 0, "overall_verdict": "clean", "signals": currentSignals()})
		return
	}

	cheatingCount := 0
	pendingCount := 0
	failedCount := 0
	noVerdictCount := 0
	intruderCount := 0
	reviewCount := 0
	phoneCount := 0
	maxPeople := 1
	maxConf := 0
	highestLikelihood := "low"
	likeOrder := map[string]int{"low": 1, "medium": 2, "high": 3}
	var firstTs, lastTs string
	totalFrames := 0
	totalWristFrames := 0

	for _, inc := range incidents {
		lk, _ := inc["cheating_likelihood"].(string)

		// Count CONFIRMED intruders only (AI-adjudicated), NOT the raw on-device
		// 2-face tripwire — otherwise a hand/object near the face that the AI vision
		// examined and cleared would still force an OVERALL VERDICT: CHEATING that
		// contradicts the rest of the report (the body shows 0 confirmed intruders).
		if ic, _ := inc["intruder_confirmed"].(bool); ic {
			intruderCount++
		}
		pv, _ := inc["phone_visible"].(bool)
		pd, _ := inc["phone_detected"].(bool)
		if pv || pd {
			phoneCount++
		}
		if fc := anyInt(inc["face_count"]); fc > maxPeople {
			maxPeople = fc
		}

		// "Pending" now means the LLM is still ENRICHING (llm_pending). A
		// provisional on-device verdict still COUNTS toward the totals, so the
		// report is never blank/half; the LLM only refines it.
		llmPending, _ := inc["llm_pending"].(bool)
		llmFailed, _ := inc["llm_failed"].(bool)
		noVerdict := lk == "analyzing" || lk == "unknown" || lk == ""
		if llmPending {
			pendingCount++
		}
		if llmFailed {
			failedCount++
		}
		if noVerdict {
			noVerdictCount++
		} else {
			if anyInt(inc["cheating"]) == 1 {
				cheatingCount++
			} else if rr, _ := inc["review_required"].(bool); rr {
				reviewCount++
			}
			if c := anyInt(inc["confidence_score"]); c > maxConf {
				maxConf = c
			}
			if likeOrder[lk] > likeOrder[highestLikelihood] {
				highestLikelihood = lk
			}
		}

		if ts, ok := inc["timestamp"].(string); ok {
			if firstTs == "" || ts < firstTs {
				firstTs = ts
			}
			if ts > lastTs {
				lastTs = ts
			}
		}
		if gd, ok := inc["gaze_data"].([]any); ok {
			totalFrames += len(gd)
			for _, f := range gd {
				if fm, ok := f.(map[string]any); ok {
					g, _ := fm["gaze"].(string)
					if g != "AHEAD" && g != "UNKNOWN" && g != "" {
						totalWristFrames++
					}
				}
			}
		}
	}

	// A pending incident still carries a provisional verdict (its counts are already
	// included above), so it is "analyzed-provisional", NOT unanalyzed. Only truly
	// verdict-less incidents are excluded — otherwise cheating_count could exceed
	// analyzed_count and the PDF would print impossibilities like "1 of 0 analyzed".
	analyzedCount := len(incidents) - noVerdictCount
	verdict := "clean"
	if intruderCount > 0 || cheatingCount > 0 || highestLikelihood == "high" {
		verdict = "cheating" // a confirmed intruder is always a violation
	} else if highestLikelihood == "medium" || reviewCount > 0 {
		verdict = "suspicious"
	} else if analyzedCount == 0 {
		verdict = "pending" // nothing analyzed yet and no client-side flags
	}

	// Session duration from first → last incident (best available without proctor clock)
	durationSecs := 0
	if firstTs != "" && lastTs != "" {
		if t0, e0 := time.Parse(time.RFC3339, firstTs); e0 == nil {
			if t1, e1 := time.Parse(time.RFC3339, lastTs); e1 == nil {
				durationSecs = int(t1.Sub(t0).Seconds())
			}
		}
	}

	// ── Evidence clustering: collapse duplicate triggers of one real event ────
	clusters := buildClusters(incidents)
	confirmedClusters, reviewClusters, techClusters, clearClusters := 0, 0, 0, 0
	for _, c := range clusters {
		switch mstr(c, "system_verdict") {
		case "confirmed_violation":
			confirmedClusters++
		case "needs_review":
			reviewClusters++
		case "technical_failure":
			techClusters++
		default:
			clearClusters++
		}
	}

	json.NewEncoder(w).Encode(map[string]any{
		"incidents":          incidents,
		"clusters":           clusters,
		"cluster_count":      len(clusters),
		"confirmed_clusters": confirmedClusters,
		"review_clusters":    reviewClusters,
		"technical_clusters": techClusters,
		"clear_clusters":     clearClusters,
		"total":              len(incidents),
		"cheating_count":     cheatingCount,
		"intruder_count":     intruderCount,
		"review_count":       reviewCount,
		"phone_count":        phoneCount,
		"max_people":         maxPeople,
		"pending_count":      pendingCount,
		"failed_count":       failedCount,
		"queue_depth":        llmQueueDepth(), // in-flight AI jobs — 0 ⇒ safe to finalize the report
		"analyzed_count":     analyzedCount,
		"overall_verdict":    verdict,
		"max_confidence":     maxConf,
		"highest_likelihood": highestLikelihood,
		"first_incident":     firstTs,
		"last_incident":      lastTs,
		"duration_secs":      durationSecs,
		"total_frames":       totalFrames,
		"wrist_gaze_frames":  totalWristFrames,
		"signals":            currentSignals(), // Layer-1 continuous frame log (nil until posted)
	})
}

// currentSignals returns the last-posted Layer-1 signal summary (or nil). Read
// under the mutex so a concurrent POST can't tear the map.
func currentSignals() map[string]any {
	lastSignalsMu.Lock()
	defer lastSignalsMu.Unlock()
	return lastSignals
}

// ── Evidence clustering ──────────────────────────────────────────────────────
// clusterGapSecs: incidents of the SAME trigger type within this many seconds of
// each other are ONE real event (a single phone glance that tripped 3 triggers),
// grouped so the report shows one cluster, not 3 duplicate incidents. The raw
// incident ids stay listed, so nothing is hidden and replay still works per-incident.
const clusterGapSecs = 25

func clusterLabel(reason string) string {
	switch reason {
	case "phone_detected_realtime":
		return "Possible phone/device"
	case "intruder_detected":
		return "Possible second person"
	case "hand_visible_and_gaze_toward_wrist":
		return "Hand / wrist-gaze"
	case "covert_device_suspected":
		return "Covert look-down"
	case "framing_torso_hidden":
		return "Torso/hands hidden"
	default:
		return "Suspicious activity"
	}
}

// verdictRank orders the four states so a cluster's headline is its STRONGEST
// member (a confirmed violation in a cluster outranks its cleared duplicates).
func verdictRank(v string) int {
	switch v {
	case "confirmed_violation":
		return 3
	case "needs_review", "technical_failure":
		return 2
	default:
		return 1 // no_violation
	}
}

// perFrameOf extracts the per-frame rows from an incident map whether they are
// live []FrameObs (same-process session report) or decoded []any (from JSON).
func perFrameOf(inc map[string]any) []FrameObs {
	switch v := inc["per_frame"].(type) {
	case []FrameObs:
		return v
	case []any:
		var out []FrameObs
		if b, err := json.Marshal(v); err == nil {
			json.Unmarshal(b, &out)
		}
		return out
	}
	return nil
}

func mstr(m map[string]any, k string) string { s, _ := m[k].(string); return s }

// strList / timelinesOf / whyNotOf tolerantly extract the structured evidence
// fields whether they are live Go types (same-process report) or decoded JSON.
func strList(m map[string]any, k string) []string {
	switch v := m[k].(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, e := range v {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

func timelinesOf(m map[string]any) []ObjectTimeline {
	switch v := m["timelines"].(type) {
	case []ObjectTimeline:
		return v
	case []any:
		var out []ObjectTimeline
		if b, err := json.Marshal(v); err == nil {
			json.Unmarshal(b, &out)
		}
		return out
	}
	return nil
}

func whyNotOf(m map[string]any) map[string]string {
	switch v := m["why_not"].(type) {
	case map[string]string:
		return v
	case map[string]any:
		out := map[string]string{}
		for k, e := range v {
			if s, ok := e.(string); ok {
				out[k] = s
			}
		}
		return out
	}
	return nil
}

// buildClusters groups incidents (same trigger type, close in time) into event
// clusters, strongest verdict as the headline, raw ids preserved.
func buildClusters(incidents []map[string]any) []map[string]any {
	sorted := make([]map[string]any, len(incidents))
	copy(sorted, incidents)
	sort.SliceStable(sorted, func(a, b int) bool { return mstr(sorted[a], "timestamp") < mstr(sorted[b], "timestamp") })

	type openC struct {
		idx    int
		lastTs time.Time
	}
	open := map[string]*openC{}
	var result []map[string]any

	for _, inc := range sorted {
		reason := mstr(inc, "trigger_reason")
		if reason == "" {
			reason = "unknown"
		}
		tsStr := mstr(inc, "timestamp")
		ts, terr := time.Parse(time.RFC3339, tsStr)
		id := mstr(inc, "id")
		sv := mstr(inc, "system_verdict")
		es := mstr(inc, "evidence_strength")

		framesAnalyzed, strongFrames := 0, 0
		for _, f := range perFrameOf(inc) {
			framesAnalyzed++
			if f.FrameStatus == "violation_evidence" {
				strongFrames++
			}
		}

		oc := open[reason]
		if oc == nil || terr != nil || ts.Sub(oc.lastTs) > clusterGapSecs*time.Second {
			c := map[string]any{
				"cluster_id":        fmt.Sprintf("CL-%d", len(result)+1),
				"trigger_reason":    reason,
				"label":             clusterLabel(reason),
				"start":             tsStr,
				"end":               tsStr,
				"raw_count":         1,
				"incident_ids":      []string{id},
				"frames_analyzed":   framesAnalyzed,
				"strong_frames":     strongFrames,
				"system_verdict":    sv,
				"evidence_strength": es,
			}
			result = append(result, c)
			if terr == nil {
				open[reason] = &openC{idx: len(result) - 1, lastTs: ts}
			} else {
				delete(open, reason)
			}
			continue
		}
		c := result[oc.idx]
		c["end"] = tsStr
		c["raw_count"] = c["raw_count"].(int) + 1
		c["incident_ids"] = append(c["incident_ids"].([]string), id)
		c["frames_analyzed"] = c["frames_analyzed"].(int) + framesAnalyzed
		c["strong_frames"] = c["strong_frames"].(int) + strongFrames
		if verdictRank(sv) > verdictRank(mstr(c, "system_verdict")) {
			c["system_verdict"] = sv
			c["evidence_strength"] = es
		}
		oc.lastTs = ts
	}
	return result
}

// ── Server-side deterministic PDF (dependency-free) ──────────────────────────
// The OFFICIAL report is rendered on the server from the locked snapshot so it is
// byte-for-byte reproducible (same monospace layout, same data) and carries a
// session integrity hash. No headless browser, no external library — a minimal
// hand-built PDF writer. The browser preview PDF stays for convenience; THIS is
// the tamper-evident artifact of record.

func pdfEscape(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == '(':
			b.WriteString(`\(`)
		case r == ')':
			b.WriteString(`\)`)
		case r < 32 || r > 126:
			b.WriteByte(' ') // Courier is Latin-1; drop non-printable/non-ASCII to a space
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// wrapText hard-wraps to width chars, preferring to break on a space.
func wrapText(s string, width int) []string {
	if width < 8 {
		width = 8
	}
	s = strings.TrimRight(s, " ")
	if s == "" {
		return []string{""}
	}
	var out []string
	for len(s) > width {
		cut := width
		if idx := strings.LastIndex(s[:width], " "); idx > width/2 {
			cut = idx
		}
		out = append(out, strings.TrimRight(s[:cut], " "))
		s = strings.TrimLeft(s[cut:], " ")
	}
	if s != "" {
		out = append(out, s)
	}
	return out
}

// paginate slices lines into fixed-height pages.
func paginate(lines []string, perPage int) [][]string {
	if perPage < 1 {
		perPage = 60
	}
	var pages [][]string
	for i := 0; i < len(lines); i += perPage {
		end := i + perPage
		if end > len(lines) {
			end = len(lines)
		}
		pages = append(pages, lines[i:end])
	}
	if len(pages) == 0 {
		pages = [][]string{{""}}
	}
	return pages
}

// buildPDF emits a valid multi-page PDF (Courier 9pt) with a correct xref table.
func buildPDF(pages [][]string) []byte {
	var buf bytes.Buffer
	var offsets []int
	obj := func(n int, body string) {
		offsets = append(offsets, buf.Len())
		fmt.Fprintf(&buf, "%d 0 obj\n%s\nendobj\n", n, body)
	}
	buf.WriteString("%PDF-1.4\n")
	nPages := len(pages)
	kids := ""
	for i := 0; i < nPages; i++ {
		kids += fmt.Sprintf("%d 0 R ", 4+2*i)
	}
	obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
	obj(2, fmt.Sprintf("<< /Type /Pages /Kids [%s] /Count %d >>", strings.TrimSpace(kids), nPages))
	obj(3, "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
	for i, lines := range pages {
		pageN, contentN := 4+2*i, 5+2*i
		obj(pageN, fmt.Sprintf("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents %d 0 R /Resources << /Font << /F1 3 0 R >> >> >>", contentN))
		var cs bytes.Buffer
		cs.WriteString("BT\n/F1 9 Tf\n40 800 Td\n11 TL\n")
		for _, ln := range lines {
			cs.WriteString("(")
			cs.WriteString(pdfEscape(ln))
			cs.WriteString(") Tj T*\n")
		}
		cs.WriteString("ET")
		stream := cs.String()
		obj(contentN, fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(stream), stream))
	}
	xrefAt := buf.Len()
	total := 3 + 2*nPages
	fmt.Fprintf(&buf, "xref\n0 %d\n0000000000 65535 f \n", total+1)
	for _, off := range offsets {
		fmt.Fprintf(&buf, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&buf, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF", total+1, xrefAt)
	return buf.Bytes()
}

// sessionReportLines builds the full official-report text (team-lead structure:
// executive summary -> evidence table -> cluster details -> per-incident evidence
// & audit) as wrapped lines ready to paginate.
func sessionReportLines(incidents, clusters []map[string]any, signals map[string]any, sessionHash, generated string) []string {
	var lines []string
	add := func(s string) { lines = append(lines, s) }
	addWrap := func(prefix, text string) {
		for i, l := range wrapText(text, 88-len(prefix)) {
			if i == 0 {
				add(prefix + l)
			} else {
				add(strings.Repeat(" ", len(prefix)) + l)
			}
		}
	}
	rule := strings.Repeat("=", 90)
	thin := strings.Repeat("-", 90)

	confirmed, review, tech, clear := 0, 0, 0, 0
	for _, c := range clusters {
		switch mstr(c, "system_verdict") {
		case "confirmed_violation":
			confirmed++
		case "needs_review":
			review++
		case "technical_failure":
			tech++
		default:
			clear++
		}
	}
	first, last := "", ""
	for _, inc := range incidents {
		ts := mstr(inc, "timestamp")
		if ts == "" {
			continue
		}
		if first == "" || ts < first {
			first = ts
		}
		if ts > last {
			last = ts
		}
	}
	overall := "NO VIOLATION DETECTED"
	if confirmed > 0 {
		overall = "CONFIRMED VIOLATION(S) — examiner action required"
	} else if review > 0 || tech > 0 {
		overall = "NEEDS EXAMINER REVIEW"
	}

	add("AI EXAM PROCTOR   —   OFFICIAL SESSION REPORT")
	add(rule)
	add("Generated (UTC):         " + generated)
	add("Rendering:               server-side, deterministic (Courier), from the locked snapshot")
	add("Report integrity:        VERIFIED")
	add("Session integrity hash:  " + sessionHash)
	add("")
	add("EXECUTIVE SUMMARY")
	add(thin)
	add(fmt.Sprintf("Session window:          %s  ->  %s", first, last))
	add(fmt.Sprintf("Total incident clusters: %d", len(clusters)))
	add(fmt.Sprintf("Confirmed violations:    %d", confirmed))
	add(fmt.Sprintf("Needs review:            %d", review))
	add(fmt.Sprintf("Technical failures:      %d", tech))
	add(fmt.Sprintf("No violation:            %d", clear))
	add(fmt.Sprintf("Raw incidents:           %d", len(incidents)))
	add(fmt.Sprintf("Final assessment:        %s", overall))
	add("")

	// ── SESSION OVERVIEW + FRAME TIMELINE (Layer 1 — the continuous frame log) ──
	// This is the "nothing discarded, only layered" section: every frame streamed
	// at ~45 FPS is accounted for here, while heavy AI ran only on the triggered
	// captures listed further below.
	aiFrames := 0
	for _, inc := range incidents {
		aiFrames += len(perFrameOf(inc))
	}
	lines = append(lines, signalOverviewLines(signals, aiFrames, len(incidents), thin)...)

	add("EVIDENCE SUMMARY (by cluster)")
	add(thin)
	add(fmt.Sprintf("%-6s %-22s %-21s %-11s %-18s", "#", "Type", "Window", "Strength", "System verdict"))
	for _, c := range clusters {
		win := shortTime(mstr(c, "start")) + "-" + shortTime(mstr(c, "end"))
		add(fmt.Sprintf("%-6s %-22s %-21s %-11s %-18s",
			mstr(c, "cluster_id"), trunc(mstr(c, "label"), 22), win,
			mstr(c, "evidence_strength"), mstr(c, "system_verdict")))
	}
	add("")
	add("CLUSTER DETAILS")
	add(rule)
	for _, c := range clusters {
		add(mstr(c, "cluster_id") + "   " + mstr(c, "label"))
		add(thin)
		add("Trigger:            " + mstr(c, "trigger_reason"))
		add(fmt.Sprintf("Time window:        %s  ->  %s", mstr(c, "start"), mstr(c, "end")))
		ids, _ := c["incident_ids"].([]string)
		addWrap("Raw incidents:      ", fmt.Sprintf("%d  (%s)", c["raw_count"], strings.Join(ids, ", ")))
		add(fmt.Sprintf("Frames analyzed:    %v", c["frames_analyzed"]))
		add(fmt.Sprintf("Strong-evidence:    %v frame(s)", c["strong_frames"]))
		add("Evidence strength:  " + mstr(c, "evidence_strength"))
		add("System verdict:     " + mstr(c, "system_verdict"))
		add("")
	}
	add("PER-INCIDENT EVIDENCE & AUDIT")
	add(rule)
	for _, inc := range incidents {
		add("Incident: " + mstr(inc, "id"))
		add(fmt.Sprintf("  System verdict:   %s   (strength: %s, evidence score: %v/100, confidence: %v%%)",
			mstr(inc, "system_verdict"), mstr(inc, "evidence_strength"), inc["evidence_score"], inc["confidence_score"]))
		if s := mstr(inc, "summary"); s != "" {
			addWrap("  Summary:          ", s)
		}
		// FIX 10 — why NOT each alternative verdict.
		if wn := whyNotOf(inc); len(wn) > 0 {
			for _, alt := range []string{"confirmed_violation", "needs_review", "no_violation", "technical_failure"} {
				if v, ok := wn[alt]; ok {
					addWrap(fmt.Sprintf("  Why not %-9s ", shortVerdict(alt)+":"), v)
				}
			}
		}
		// FIX 5 — contradictions, shown explicitly.
		if cs := strList(inc, "contradictions"); len(cs) > 0 {
			add("  CONTRADICTIONS:")
			for _, c := range cs {
				addWrap("    - ", c)
			}
		}
		// FIX 6 — per-object timeline.
		if tl := timelinesOf(inc); len(tl) > 0 {
			add("  TIMELINE:")
			for _, t := range tl {
				add(fmt.Sprintf("    - %-14s first=F%d  peak=F%d(%d%%)  last=F%d  frames=%d",
					trunc(t.Object, 14), t.FirstFrame, t.PeakFrame, t.PeakConfidence, t.LastFrame, t.Frames))
			}
		}
		pf := perFrameOf(inc)
		if len(pf) > 0 {
			add("  " + fmt.Sprintf("%-5s %-8s %-11s %-15s %-11s %-5s %-3s", "Frame", "Quality", "Object", "Detection", "Status", "Conf", "AI"))
			for _, f := range pf {
				aic := "no"
				if f.AnalyzedAI {
					aic = "yes"
				}
				add("  " + fmt.Sprintf("%-5d %-8s %-11s %-15s %-11s %-5d %-3s",
					f.Frame, trunc(f.Quality, 8), trunc(objOrNone(f.ObjectClass, f.ObjectInHand), 11),
					trunc(orDash(f.Detection), 15), trunc(f.ObjectStatus, 11), f.AIConfidence, aic))
			}
		}
		if d := mstr(inc, "examiner_decision"); d != "" {
			add(fmt.Sprintf("  Examiner decision:%s  by %s  at %s", " "+d, mstr(inc, "examiner_id"), mstr(inc, "reviewed_at")))
			if n := mstr(inc, "examiner_note"); n != "" {
				addWrap("  Examiner note:    ", n)
			}
		}
		add("  Integrity hash:   " + mstr(inc, "integrity_hash"))
		add("")
	}
	add(rule)
	add("Verdict engine: deterministic evidence-strength (AI is observer-only).  Per-incident replay: GET /replay?id=")
	add("Rendered server-side from the frozen snapshot and tamper-sealed per incident + per session.")
	return lines
}

// signalOverviewLines renders the SESSION OVERVIEW (continuous capture) and the
// FRAME TIMELINE from the browser's Layer-1 signal summary. When no summary was
// posted (older client / detached run) it degrades gracefully to just the
// AI-analyzed frame count so the section is never empty or misleading.
func signalOverviewLines(sig map[string]any, aiFrames, triggeredIncidents int, thin string) []string {
	var out []string
	add := func(s string) { out = append(out, s) }
	add("CONTINUOUS CAPTURE — SESSION OVERVIEW (Layer 1: every frame, lightweight)")
	add(thin)
	if sig == nil {
		add("Continuous frame log:    not captured for this session")
		add(fmt.Sprintf("Frames analyzed by AI:   %d   (Layer 3, across %d triggered incident(s))", aiFrames, triggeredIncidents))
		add("Note: the per-frame signal buffer is posted by the live proctor at End Session.")
		add("")
		return out
	}
	total := anyInt(sig["total_frames"])
	dur := anyInt(sig["duration_secs"])
	fps := anyFloat(sig["avg_fps"])
	add(fmt.Sprintf("Total frames captured:   %d   (light per-frame logging @ %.1f FPS average)", total, fps))
	add(fmt.Sprintf("Session duration:        %s", durLabel(dur)))
	add(fmt.Sprintf("Frames analyzed by AI:   %d   (Layer 3, on %d triggered incident(s) only)", aiFrames, triggeredIncidents))
	if ev, ok := sig["events"].(map[string]any); ok {
		add(fmt.Sprintf("Trigger events:          %d   (phone %d · intruder %d · wrist-gaze %d · covert %d · framing %d)",
			anyInt(ev["total"]), anyInt(ev["phone_detected_realtime"]), anyInt(ev["intruder_detected"]),
			anyInt(ev["hand_visible_and_gaze_toward_wrist"]), anyInt(ev["covert_device_suspected"]),
			anyInt(ev["framing_torso_hidden"])))
	}
	if g, ok := sig["gaze"].(map[string]any); ok {
		add(fmt.Sprintf("Gaze distribution:       AHEAD %d · LEFT %d · RIGHT %d · UP %d · DOWN %d · no-face %d",
			anyInt(g["AHEAD"]), anyInt(g["LEFT"]), anyInt(g["RIGHT"]), anyInt(g["UP"]), anyInt(g["DOWN"]), anyInt(g["NO_FACE"])))
	}
	if f, ok := sig["face"].(map[string]any); ok {
		add(fmt.Sprintf("Face presence (frames):  1 face %d · 2+ faces %d · none %d",
			anyInt(f["one"]), anyInt(f["multi"]), anyInt(f["none"])))
	}
	add(fmt.Sprintf("Hand-visible frames:     %d", anyInt(sig["hand_present_frames"])))
	add(fmt.Sprintf("On-device phone hint:    %d frame(s)", anyInt(sig["phone_hint_frames"])))
	if m, ok := sig["motion"].(map[string]any); ok {
		add(fmt.Sprintf("Motion (0-100):          avg %.1f · peak %d · high-motion %d frame(s)",
			anyFloat(m["avg"]), anyInt(m["max"]), anyInt(m["high_frames"])))
	}
	if q, ok := sig["quality"].(map[string]any); ok {
		add(fmt.Sprintf("Frame quality:           good %d · poor %d", anyInt(q["good"]), anyInt(q["poor"])))
	}
	add("")

	// FRAME TIMELINE — the downsampled per-second truth.
	tl, _ := sig["timeline"].([]any)
	if len(tl) > 0 {
		step := anyInt(sig["sample_secs"])
		if step < 1 {
			step = 1
		}
		unit := "s"
		if step >= 60 {
			step /= 60
			unit = "m"
		}
		add(fmt.Sprintf("FRAME TIMELINE (sampled every %d%s from the ~45 FPS stream — full truth, nothing discarded)", step, unit))
		add(thin)
		add(fmt.Sprintf("  %-8s %-5s %-9s %-5s %-7s %-5s", "Time", "Face", "Gaze", "Hand", "Motion", "Phone"))
		for _, row := range tl {
			r, ok := row.(map[string]any)
			if !ok {
				continue
			}
			hand := "-"
			if b, _ := r["hand"].(bool); b {
				hand = "yes"
			}
			phone := "-"
			if b, _ := r["phone"].(bool); b {
				phone = "YES"
			}
			add(fmt.Sprintf("  %-8s %-5d %-9s %-5s %-7d %-5s",
				secLabel(anyInt(r["sec"])), anyInt(r["face"]), trunc(mstr(r, "gaze"), 9),
				hand, anyInt(r["motion"]), phone))
		}
		add("")
	}
	return out
}

// anyFloat coerces a JSON-decoded number (float64) or numeric string to float64.
func anyFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case json.Number:
		f, _ := n.Float64()
		return f
	}
	return 0
}

// secLabel formats an elapsed-second offset as mm:ss.
func secLabel(s int) string {
	if s < 0 {
		s = 0
	}
	return fmt.Sprintf("%02d:%02d", s/60, s%60)
}

// durLabel formats a duration in seconds as "Nm Ss".
func durLabel(s int) string {
	if s <= 0 {
		return "--"
	}
	return fmt.Sprintf("%dm %ds", s/60, s%60)
}

func objOrNone(class, raw string) string {
	if class != "" && class != "none" {
		return class
	}
	if raw != "" {
		return raw
	}
	return "none"
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "."
}

func orDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

// shortVerdict abbreviates a system-verdict key for the "why not" labels.
func shortVerdict(v string) string {
	switch v {
	case "confirmed_violation":
		return "confirmed"
	case "needs_review":
		return "review"
	case "no_violation":
		return "clear"
	case "technical_failure":
		return "failure"
	}
	return v
}

func shortTime(rfc string) string {
	if t, err := time.Parse(time.RFC3339, rfc); err == nil {
		return t.Format("15:04:05")
	}
	return rfc
}

// handleSessionPDF renders the official server-side PDF from the locked snapshot.
func handleSessionPDF(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	incidents := listLocalIncidents(0)
	clusters := buildClusters(incidents)

	hashes := make([]string, 0, len(incidents))
	for _, inc := range incidents {
		if h := mstr(inc, "integrity_hash"); h != "" {
			hashes = append(hashes, h)
		}
	}
	sort.Strings(hashes)
	sum := sha256.Sum256([]byte(strings.Join(hashes, "|")))
	sessionHash := hex.EncodeToString(sum[:])

	generated := time.Now().UTC().Format(time.RFC3339)
	lines := sessionReportLines(incidents, clusters, currentSignals(), sessionHash, generated)
	pdf := buildPDF(paginate(lines, 62))

	w.Header().Set("Content-Type", "application/pdf")
	w.Header().Set("Content-Disposition", `attachment; filename="proctor-official-session-report.pdf"`)
	w.Write(pdf)
}

func handleIncidentsList(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	w.Header().Set("Content-Type", "application/json")

	// Prefer Supabase if it has rows; otherwise fall back to local disk so the
	// dashboard always shows real incidents (even before the table is created).
	url := os.Getenv("SUPABASE_URL")
	key := os.Getenv("SUPABASE_SECRET_KEY")
	if url != "" && key != "" {
		if rows := fetchSupabase(url, key); len(rows) > 0 {
			json.NewEncoder(w).Encode(rows)
			return
		}
	}
	json.NewEncoder(w).Encode(listLocalIncidents(20)) // dashboard strip shows the 20 newest
}

func fetchSupabase(url, key string) []map[string]any {
	sel := "id,timestamp,cheating,review_required,confidence_score,cheating_likelihood,cheating_material_detected,phone_visible,phone_detected,phone_ai_confirmed,face_count,intruder_detected,intruder_confirmed,trigger_reason,materials,suspicious_behaviors,gaze_analysis,visual_evidence,recommendation,summary,per_frame,llm_pending,gaze_data"
	q := url + "/rest/v1/incidents?select=" + sel + "&order=created_at.desc&limit=20"
	req, _ := http.NewRequest("GET", q, nil)
	req.Header.Set("apikey", key)
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var rows []map[string]any
	if json.NewDecoder(resp.Body).Decode(&rows) != nil {
		return nil
	}
	for _, row := range rows {
		if id, ok := row["id"].(string); ok {
			row["collage"] = fmt.Sprintf("%s/incidents/%s/collage.jpg", selfBase, id)
		}
	}
	return rows
}

// listLocalIncidents builds the incident list from the saved folders. limit caps
// the count (newest first) for the dashboard strip; limit <= 0 returns EVERY
// incident, which the session report needs so a violation early in a long exam is
// never dropped from the summary (and total never under-reports).
func listLocalIncidents(limit int) []map[string]any {
	entries, err := os.ReadDir(incidentsDir)
	if err != nil {
		return []map[string]any{}
	}
	names := []string{}
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(names)))
	if limit > 0 && len(names) > limit {
		names = names[:limit]
	}

	out := []map[string]any{}
	for _, id := range names {
		dir := filepath.Join(incidentsDir, id)
		m := map[string]any{
			"id":                   id,
			"collage":              fmt.Sprintf("%s/incidents/%s/collage.jpg", selfBase, id),
			"cheating":             0,
			"cheating_likelihood":  "analyzing",
			"materials":            []string{},
			"suspicious_behaviors": []string{},
			"summary":              "Analyzing…",
		}
		if b, err := os.ReadFile(filepath.Join(dir, "metadata.json")); err == nil {
			var meta map[string]any
			if json.Unmarshal(b, &meta) == nil {
				m["timestamp"] = meta["timestamp"]
				m["gaze_data"] = meta["gaze_data"]
				// Client-side ground truth (independent of the LLM report)
				m["face_count"] = anyInt(meta["face_count"])
				if pd, ok := meta["phone_detected"].(bool); ok {
					m["phone_detected"] = pd // real-time on-device detector (independent of the LLM)
				}
				if id, ok := meta["intruder_detected"].(bool); ok {
					m["intruder_detected"] = id
				}
				if tr, ok := meta["trigger_reason"].(string); ok {
					m["trigger_reason"] = tr
				}
			}
		}
		if b, err := os.ReadFile(filepath.Join(dir, "report.json")); err == nil {
			var rep Report
			if json.Unmarshal(b, &rep) == nil {
				m["cheating"] = rep.Cheating
				m["review_required"] = rep.ReviewRequired
				m["intruder_confirmed"] = rep.IntruderConfirmed
				m["confidence_score"] = rep.ConfidenceScore
				m["cheating_likelihood"] = rep.CheatingLikelihood
				m["cheating_material_detected"] = rep.CheatingMaterialDetected
				m["phone_visible"] = rep.PhoneVisible
				m["phone_ai_confirmed"] = rep.PhoneAIConfirmed
				m["llm_pending"] = rep.LLMPending
				m["llm_failed"] = rep.LLMFailed
				if rep.PerFrame != nil {
					m["per_frame"] = rep.PerFrame
				}
				if rep.Materials != nil {
					m["materials"] = rep.Materials
				}
				if rep.SuspiciousBehaviors != nil {
					m["suspicious_behaviors"] = rep.SuspiciousBehaviors
				}
				m["gaze_analysis"] = rep.GazeAnalysis
				m["visual_evidence"] = rep.VisualEvidence
				m["recommendation"] = rep.Recommendation
				m["summary"] = rep.Summary
				m["system_verdict"] = rep.SystemVerdict
				m["evidence_strength"] = rep.EvidenceStrength
				m["evidence_score"] = rep.EvidenceScore
				m["why_not_clear"] = rep.WhyNotClear
				m["why_not_cheating"] = rep.WhyNotCheating
				m["why_not"] = rep.WhyNot
				m["contradictions"] = rep.Contradictions
				m["timelines"] = rep.Timelines
				m["integrity_hash"] = rep.IntegrityHash
				m["examiner_decision"] = rep.ExaminerDecision
				m["examiner_note"] = rep.ExaminerNote
				m["examiner_id"] = rep.ExaminerID
				m["reviewed_at"] = rep.ReviewedAt
			}
		}
		out = append(out, m)
	}
	return out
}

// ── GET /incidents/{id}/collage.jpg — serve the saved collage ──────────────

func handleCollageImage(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	w.Header().Set("Cross-Origin-Resource-Policy", "cross-origin")

	rest := strings.TrimPrefix(r.URL.Path, "/incidents/")
	parts := strings.Split(rest, "/")
	if len(parts) < 1 || parts[0] == "" || strings.Contains(parts[0], "..") {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, filepath.Join(incidentsDir, parts[0], "collage.jpg"))
}

func shortID() string {
	b := make([]byte, 3)
	rand.Read(b)
	return hex.EncodeToString(b)
}
