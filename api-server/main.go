package main

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var geminiClient = &http.Client{Timeout: 120 * time.Second}

// geminiMu serialises all Gemini calls — free tier is 20 RPM so one-at-a-time
// with built-in retry-after handling is the safest strategy.
var geminiMu   sync.Mutex
var geminiNext time.Time // earliest time the next call is allowed

const port = ":8080"
const incidentsDir = "incidents"
const geminiURL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
const selfBase = "http://localhost:8080"

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
	IntruderConfirmed        bool     `json:"intruder_confirmed"`
	Materials                []string `json:"materials"`
	SuspiciousBehaviors      []string `json:"suspicious_behaviors"`
	GazeAnalysis             string   `json:"gaze_analysis"`
	VisualEvidence           string   `json:"visual_evidence"`
	Recommendation           string   `json:"recommendation"`
	Summary                  string   `json:"summary"`
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
	FaceCount                int              `json:"face_count"`
	IntruderDetected         bool             `json:"intruder_detected"`
	TriggerReason            string           `json:"trigger_reason"`
	Materials                []string         `json:"materials"`
	SuspiciousBehaviors      []string         `json:"suspicious_behaviors"`
	GazeAnalysis             string           `json:"gaze_analysis"`
	VisualEvidence           string           `json:"visual_evidence"`
	Recommendation           string           `json:"recommendation"`
	Summary                  string           `json:"summary"`
	GazeData                 []map[string]any `json:"gaze_data"`
	Collage                  string           `json:"collage"`
}

// ── Gemini API types ───────────────────────────────────────────────────────

type geminiRequest struct {
	Contents         []geminiContent `json:"contents"`
	GenerationConfig *genConfig      `json:"generationConfig,omitempty"`
}
type genConfig struct {
	ResponseMimeType string        `json:"responseMimeType"`
	ThinkingConfig   *thinkingCfg  `json:"thinkingConfig,omitempty"`
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
	time.Sleep(2 * time.Second)
	groqKey   := os.Getenv("GROQ_API_KEY")
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
		var report Report
		var raw string
		var callErr error
		if groqKey != "" {
			fmt.Printf("[reprocess] %s — calling Groq…\n", j.id)
			report, raw, callErr = callGroq(groqKey, imageURLs, j.id, timestamp, frames, phoneDetected, handPosture, faceCount, intruderDetected, triggerReason)
		} else {
			fmt.Printf("[reprocess] %s — calling Gemini…\n", j.id)
			report, raw, callErr = callGemini(geminiKey, imageURLs, j.id, timestamp, frames, phoneDetected, handPosture, faceCount, intruderDetected, triggerReason)
		}
		if callErr != nil {
			fmt.Printf("[reprocess] %s — error: %v\n", j.id, callErr)
			continue
		}
		os.WriteFile(filepath.Join(j.dir, "report.json"), []byte(raw), 0644)
		fmt.Printf("[reprocess] %s — done: cheating=%d conf=%d%% (%s)\n",
			j.id, report.Cheating, report.ConfidenceScore, report.CheatingLikelihood)
	}
}

// ── Main ───────────────────────────────────────────────────────────────────

func main() {
	loadEnv(".env")
	os.MkdirAll(incidentsDir, 0755)

	if os.Getenv("GROQ_API_KEY") != "" {
		fmt.Printf("Vision LLM: Groq (%s) — ENABLED\n", groqModel)
		go reprocessOrphans()
	} else if os.Getenv("GEMINI_API_KEY") != "" {
		fmt.Println("Vision LLM: Gemini 2.0 Flash — ENABLED")
		go reprocessOrphans()
	} else {
		fmt.Println("Vision LLM: DISABLED (set GROQ_API_KEY or GEMINI_API_KEY)")
	}
	if os.Getenv("SUPABASE_URL") != "" && os.Getenv("SUPABASE_SECRET_KEY") != "" {
		fmt.Println("Supabase persistence: ENABLED")
	} else {
		fmt.Println("Supabase persistence: DISABLED (set SUPABASE_URL + SUPABASE_SECRET_KEY)")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/incident", handleIncident)         // POST one incident
	mux.HandleFunc("/incidents", handleIncidentsList)   // GET list (dashboard)
	mux.HandleFunc("/incidents/", handleCollageImage)   // GET /incidents/{id}/collage.jpg
	mux.HandleFunc("/session-report", handleSessionReport) // GET full session PDF data
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
		"incident_id":    incidentID,
		"timestamp":      req.Timestamp,
		"frame_count":    saved,
		"gaze_labels":    req.GazeLabels,
		"gaze_data":         frames,
		"phone_detected":    req.PhoneDetected,
		"hand_posture":      req.HandPosture,
		"face_count":        req.FaceCount,
		"intruder_detected": req.IntruderDetected,
		"trigger_reason":    req.TriggerReason,
	}
	if b, err := json.MarshalIndent(meta, "", "  "); err == nil {
		os.WriteFile(dir+"/metadata.json", b, 0644)
	}

	// LLM + Supabase run async — never block the capture loop.
	apiKey := os.Getenv("GEMINI_API_KEY")
	reportWanted := (os.Getenv("GROQ_API_KEY") != "" || apiKey != "") && len(req.Frames) > 0
	go processIncident(incidentID, dir, req, frames, apiKey)

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

// processIncident: call vision LLM (Groq or Gemini), then persist to Supabase.
func processIncident(incidentID, dir string, req IncidentRequest, frames []map[string]any, apiKey string) {
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
		FaceCount:           req.FaceCount,
		IntruderDetected:    req.IntruderDetected,
		TriggerReason:       req.TriggerReason,
	}

	// Send the individual full-res frames (the 5 most time-spread of the 6) to
	// the vision model — far more legible for small objects than a downscaled
	// collage. The collage is kept only as the dashboard thumbnail.
	imageURLs := selectFrameURLs(req.Frames, maxLLMImages)
	if len(imageURLs) > 0 {
		var report Report
		var raw string
		var err error

		groqKey := os.Getenv("GROQ_API_KEY")
		if groqKey != "" {
			report, raw, err = callGroq(groqKey, imageURLs, incidentID, req.Timestamp, frames, req.PhoneDetected, req.HandPosture, req.FaceCount, req.IntruderDetected, req.TriggerReason)
			if err != nil {
				fmt.Printf("[incident] %s  Groq error: %v\n", incidentID, err)
			}
		} else if apiKey != "" {
			report, raw, err = callGemini(apiKey, imageURLs, incidentID, req.Timestamp, frames, req.PhoneDetected, req.HandPosture, req.FaceCount, req.IntruderDetected, req.TriggerReason)
			if err != nil {
				fmt.Printf("[incident] %s  Gemini error: %v\n", incidentID, err)
			}
		}

		if err == nil && raw != "" {
			// ── Deterministic decision engine (cross-frame consistency) ──────
			// The model only GROUNDS what it sees per frame; WE decide the
			// verdict from how consistently evidence appears. Rationale:
			//   • A hallucination is random → shows up in ~1 frame.
			//   • A real object is held for the whole second → 4–6 frames.
			//   • An intruder (2+ validated faces) is a hardware sensor fact.
			// So: object in ≥2 frames = CONFIRMED; in exactly 1 = NEEDS REVIEW
			// (flagged, not accused); none = CLEAR. Intruder is always confirmed.
			decideVerdict(&report, rec.IntruderDetected)

			// Persist the corrected report so the dashboard + on-disk file agree.
			if b, mErr := json.MarshalIndent(report, "", "  "); mErr == nil {
				raw = string(b)
			}

			os.WriteFile(dir+"/report.json", []byte(raw), 0644)
			rec.Cheating = report.Cheating
			rec.ReviewRequired = report.ReviewRequired
			rec.ConfidenceScore = report.ConfidenceScore
			rec.CheatingLikelihood = report.CheatingLikelihood
			rec.CheatingMaterialDetected = report.CheatingMaterialDetected
			rec.PhoneVisible = report.PhoneVisible
			if report.Materials != nil {
				rec.Materials = report.Materials
			}
			if report.SuspiciousBehaviors != nil {
				rec.SuspiciousBehaviors = report.SuspiciousBehaviors
			}
			rec.GazeAnalysis = report.GazeAnalysis
			rec.VisualEvidence = report.VisualEvidence
			rec.Recommendation = report.Recommendation
			rec.Summary = report.Summary
			state := "CLEAR"
			if report.Cheating == 1 {
				state = "CHEATING"
			} else if report.ReviewRequired {
				state = "NEEDS_REVIEW"
			}
			fmt.Printf("[incident] %s  verdict: %s  phone=%v intruder=%v(%d) materials=%v conf=%d%% (%s)\n",
				incidentID, state, report.PhoneVisible, rec.IntruderDetected, rec.FaceCount, report.Materials, report.ConfidenceScore, report.CheatingLikelihood)
		}
	}

	insertSupabase(rec)
}

// decideVerdict applies the cross-frame consistency rule. The model fills
// PerFrame (what it actually sees in each frame); this recomputes the verdict,
// materials, phone flag, intruder flag, evidence text and summary
// DETERMINISTICALLY — so the model can neither hallucinate a conviction nor
// bury a real object, and the report is always internally consistent.
func decideVerdict(r *Report, clientIntruder bool) {
	objFrames := 0    // frames clearly showing a real held object
	secondFrames := 0 // frames showing a second person
	typeCount := map[string]int{}
	for _, f := range r.PerFrame {
		o := strings.ToLower(strings.TrimSpace(f.ObjectInHand))
		if o != "" && o != "none" && o != "empty" && o != "hand" && o != "nothing" {
			objFrames++
			typeCount[o]++
		}
		if f.SecondPerson {
			secondFrames++
		}
	}
	// Most-frequently-seen object type wins.
	objType, best := "", 0
	for t, c := range typeCount {
		if c > best {
			best, objType = c, t
		}
	}
	// Fallback if the model skipped the per-frame table: trust its own flags
	// so a real detection is never silently dropped.
	if len(r.PerFrame) == 0 {
		if r.PhoneVisible {
			objFrames, objType = 2, "phone"
		} else if len(r.Materials) > 0 {
			objFrames, objType = 2, strings.ToLower(strings.TrimSpace(r.Materials[0]))
		}
	}

	intruder := clientIntruder || secondFrames >= 2
	confirmed := objFrames >= 2 && objType != "" && objType != "other" // persistent → real
	ambiguous := !confirmed && objFrames >= 1                          // one-frame blip → review

	// Reset convict-able fields; set them only from the rule below.
	r.PhoneVisible = false
	r.CheatingMaterialDetected = false
	r.Materials = []string{}
	r.IntruderConfirmed = intruder
	mats := []string{}

	switch {
	case confirmed:
		r.Cheating, r.ReviewRequired = 1, false
		r.CheatingMaterialDetected = true
		r.PhoneVisible = objType == "phone"
		mats = append(mats, objType)
		if r.ConfidenceScore < 85 {
			r.ConfidenceScore = 85
		}
		r.CheatingLikelihood = "high"
	case intruder:
		r.Cheating, r.ReviewRequired = 1, false
		if r.ConfidenceScore < 90 {
			r.ConfidenceScore = 90
		}
		r.CheatingLikelihood = "high"
	case ambiguous:
		r.Cheating, r.ReviewRequired = 0, true
		if r.ConfidenceScore == 0 || r.ConfidenceScore > 65 {
			r.ConfidenceScore = 50
		}
		r.CheatingLikelihood = "medium"
	default:
		r.Cheating, r.ReviewRequired = 0, false
		if r.ConfidenceScore == 0 {
			r.ConfidenceScore = 80
		}
		r.CheatingLikelihood = "low"
	}
	r.Materials = mats

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

	// Deterministic, non-contradictory one-line summary.
	switch {
	case confirmed && intruder:
		r.Summary = fmt.Sprintf("CONFIRMED: %s held across %d frames AND a second person present.", objType, objFrames)
	case confirmed:
		r.Summary = fmt.Sprintf("CONFIRMED cheating: %s clearly held across %d of %d frames.", objType, objFrames, len(r.PerFrame))
	case intruder:
		r.Summary = "CONFIRMED: a second person is present in the candidate's frame."
	case ambiguous:
		r.Summary = fmt.Sprintf("NEEDS REVIEW: a possible object appeared in only 1 frame (%s) — not consistent enough to confirm. Manual check advised.", objType)
	default:
		r.Summary = "CLEAR: no held object across the frames and no second person — normal behaviour."
	}
	if r.Recommendation == "" {
		switch {
		case r.Cheating == 1:
			r.Recommendation = "Flag this candidate for examiner action."
		case r.ReviewRequired:
			r.Recommendation = "Queue for quick manual review of the flagged frame."
		default:
			r.Recommendation = "No action required."
		}
	}
}

// ── Vision LLM — full JSON payload + binary cheating verdict ──────────────

// maxLLMImages is the per-request image cap for the vision model.
// Groq Llama 4 Scout accepts at most 5 images per request, so we capture 6
// frames (kept on disk + in the collage thumbnail) and send the 5 most
// time-spread frames to the model.
const maxLLMImages = 5

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
	idx := make([]int, max)
	for i := 0; i < max; i++ {
		idx[i] = i * (n - 1) / (max - 1)
	}
	return idx
}

// selectFrameURLs normalises capture frames to full data URLs and caps the
// count to maxLLMImages, sampling evenly across the set.
func selectFrameURLs(frames []string, max int) []string {
	out := make([]string, 0, max)
	for _, i := range pickEvenly(len(frames), max) {
		u := strings.TrimSpace(frames[i])
		if u == "" {
			continue
		}
		if !strings.HasPrefix(u, "data:") {
			u = "data:image/jpeg;base64," + u
		}
		out = append(out, u)
	}
	return out
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
- description: one short sentence on what is actually visible in this frame.

⚠ OBSERVATION RULES (critical):
- NEVER invent an object. If you do not CLEARLY see a rigid rectangular device, object_in_hand is NOT "phone". A bare hand, curled fingers, knuckles, a chin-rest, a shadow, or a watch strap is NOT a phone.
- A hand resting on the chin/cheek/mouth/temple with nothing in it = object_in_hand "none". This is normal and is the #1 false alarm — do not label it.
- "phone" = dark/glowing rigid slab. "paper" = light-colored flat sheet/page. Never swap these. If a flat light object, it is "paper", not "phone".
- Only set object_in_hand to a real type for a frame where you genuinely see it. If unsure for a frame, use "none". It is fine for different frames to differ — the rule engine wants the truth per frame, and it only confirms an object that PERSISTS across multiple frames (so a one-frame guess will NOT cause a false accusation — but inventing the same object in every frame WILL).
- Ignore the hints phone_detected_client / hand_posture="holding_object": they fire from mere curled fingers. Report object_in_hand ONLY from what you see.

Then provide overall fields. Respond with ONLY a valid JSON object — no prose, no markdown:
{
  "per_frame": [
    {"frame": 1, "hand_visible": true, "object_in_hand": "none", "second_person": false, "gaze": "ahead", "description": "..."}
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
		"incident_id":             incidentID,
		"timestamp":               timestamp,
		"trigger":                 triggerReason,
		"images_provided":         len(imageURLs),
		"frame_count":             len(frames),
		"phone_detected_client":   phoneDetected,
		"hand_posture":            handPosture,
		"face_count":              faceCount,
		"intruder_detected_client": intruderDetected,
		"frames":                  frames,
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
	for attempt := 1; attempt <= 5; attempt++ {
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
			if wait == 0 {
				wait = 20 * time.Second
			}
			fmt.Printf("[gemini] attempt %d quota error — waiting %.1fs\n", attempt, wait.Seconds())
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

var retryRe = regexp.MustCompile(`retry in ([0-9]+(?:\.[0-9]+)?)s`)

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

const groqURL   = "https://api.groq.com/openai/v1/chat/completions"
const groqModel = "meta-llama/llama-4-scout-17b-16e-instruct"

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
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL *groqImageURL   `json:"image_url,omitempty"`
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
		Temperature:    0.1, // low temp → consistent, deterministic verdicts
		MaxTokens:      2048, // room for the per-frame table
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

	req, _ := http.NewRequest("POST", groqURL, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := geminiClient.Do(req) // reuse the 120s timeout client
	if err != nil {
		return Report{}, "", err
	}
	defer resp.Body.Close()

	var gresp groqResponse
	if err := json.NewDecoder(resp.Body).Decode(&gresp); err != nil {
		return Report{}, "", err
	}
	if gresp.Error != nil {
		return Report{}, "", fmt.Errorf("groq error: %s", gresp.Error.Message)
	}
	if len(gresp.Choices) == 0 {
		return Report{}, "", fmt.Errorf("empty response from Groq")
	}

	raw := gresp.Choices[0].Message.Content
	clean := extractJSON(raw)
	var rep Report
	if err := json.Unmarshal([]byte(clean), &rep); err != nil {
		return Report{}, raw, fmt.Errorf("parse groq report: %w", err)
	}
	return rep, clean, nil
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
	req, _ := http.NewRequest("POST", url+"/rest/v1/incidents", bytes.NewReader(body))
	req.Header.Set("apikey", key)
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Prefer", "return=minimal")

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
func handleSessionReport(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	w.Header().Set("Content-Type", "application/json")

	incidents := listLocalIncidents()
	if len(incidents) == 0 {
		json.NewEncoder(w).Encode(map[string]any{"incidents": []any{}, "total": 0, "cheating_count": 0, "pending_count": 0, "overall_verdict": "clean"})
		return
	}

	cheatingCount := 0
	pendingCount  := 0
	intruderCount := 0
	reviewCount   := 0
	phoneCount    := 0
	maxPeople     := 1
	maxConf       := 0
	highestLikelihood := "low"
	likeOrder := map[string]int{"low": 1, "medium": 2, "high": 3}
	var firstTs, lastTs string
	totalFrames     := 0
	totalWristFrames := 0

	for _, inc := range incidents {
		lk, _ := inc["cheating_likelihood"].(string)

		// Client-side ground truth — counted for every incident, analyzed or not
		if id, _ := inc["intruder_detected"].(bool); id {
			intruderCount++
		}
		if pv, _ := inc["phone_visible"].(bool); pv {
			phoneCount++
		}
		if fc := anyInt(inc["face_count"]); fc > maxPeople {
			maxPeople = fc
		}

		// Skip unanalyzed incidents for verdict calculation
		if lk == "analyzing" || lk == "unknown" || lk == "" {
			pendingCount++
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

	analyzedCount := len(incidents) - pendingCount
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

	json.NewEncoder(w).Encode(map[string]any{
		"incidents":           incidents,
		"total":               len(incidents),
		"cheating_count":      cheatingCount,
		"intruder_count":      intruderCount,
		"review_count":        reviewCount,
		"phone_count":         phoneCount,
		"max_people":          maxPeople,
		"pending_count":       pendingCount,
		"analyzed_count":      analyzedCount,
		"overall_verdict":     verdict,
		"max_confidence":      maxConf,
		"highest_likelihood":  highestLikelihood,
		"first_incident":      firstTs,
		"last_incident":       lastTs,
		"duration_secs":       durationSecs,
		"total_frames":        totalFrames,
		"wrist_gaze_frames":   totalWristFrames,
	})
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
	json.NewEncoder(w).Encode(listLocalIncidents())
}

func fetchSupabase(url, key string) []map[string]any {
	sel := "id,timestamp,cheating,confidence_score,cheating_likelihood,cheating_material_detected,phone_visible,face_count,intruder_detected,trigger_reason,materials,suspicious_behaviors,gaze_analysis,visual_evidence,recommendation,summary,gaze_data"
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

// listLocalIncidents builds the dashboard list from the saved incident folders.
func listLocalIncidents() []map[string]any {
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
	if len(names) > 20 {
		names = names[:20]
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
