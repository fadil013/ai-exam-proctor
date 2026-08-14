package main

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// frames builds a PerFrame slice where each object label is one frame.
func frames(objs ...string) []FrameObs {
	out := make([]FrameObs, len(objs))
	for i, o := range objs {
		out[i] = FrameObs{Frame: i + 1, ObjectInHand: o, Gaze: "down"}
	}
	return out
}

const trig = "hand_visible_and_gaze_toward_wrist"

// ── classifyObject ─────────────────────────────────────────────────────────

func TestClassifyObject(t *testing.T) {
	cases := map[string]string{
		"":               "none",
		"none":           "none",
		"hand":           "none",
		"water bottle":   "benign",
		"a water bottle": "benign",
		"cup":            "benign",
		"coffee mug":     "benign",
		"pen":            "benign",
		"snack":          "benign",
		"phone":          "cheat",
		"cell phone":     "cheat",
		"smartphone":     "cheat",
		"paper":          "cheat",
		"notes":          "cheat",
		"book":           "cheat",
		"earbud":         "cheat",
		"smartwatch":     "cheat",
		"stapler":        "unknown",
		// Adversarial: real cheat aids the model describes as "open …" must NOT
		// be cleared by the benign word "pen" hiding inside "open".
		"open book":       "cheat",
		"open notebook":   "cheat",
		"open laptop":     "cheat",
		"laptop open":     "cheat",
		"open note":       "cheat",
		"phone open":      "cheat",
		"expensive phone": "cheat",
		"phone case":      "cheat",
		"surface tablet":  "cheat",
		// "face" inside "surface"/"facetime" must not clear a device.
		"facetime": "unknown",
		// Head-noun: a paper CUP is a drink container, not notes.
		"paper cup":  "benign",
		"coffee cup": "benign",
		// Plurals.
		"earbuds": "cheat",
		"books":   "cheat",
		"papers":  "cheat",
		"cups":    "benign",
	}
	for in, want := range cases {
		if got := classifyObject(in); got != want {
			t.Errorf("classifyObject(%q) = %q, want %q", in, got, want)
		}
	}
}

// The whole reason for whole-word matching: an OPEN BOOK across frames must
// still CONVICT (the substring bug silently cleared it).
func TestOpenBookStillCheats(t *testing.T) {
	r := &Report{PerFrame: frames("none", "open book", "open book", "none", "none", "none")}
	decideVerdict(r, false, trig, false, false, true, true)
	if r.Cheating != 1 {
		t.Fatalf("an open book across 2 frames must be CHEATING, got cheating=%d", r.Cheating)
	}
}

// ── The reported bug: a water bottle must NOT be cheating ───────────────────

func TestWaterBottleNeverCheating(t *testing.T) {
	r := &Report{PerFrame: frames("none", "water bottle", "water bottle", "water bottle", "none", "none")}
	decideVerdict(r, false, trig, false, false, true, true)
	if r.Cheating != 0 {
		t.Fatalf("water bottle flagged as cheating (Cheating=%d)", r.Cheating)
	}
	if r.ReviewRequired {
		t.Errorf("water bottle should be CLEAR, not NEEDS REVIEW")
	}
	if len(r.Materials) != 0 {
		t.Errorf("water bottle must not appear as a detected material, got %v", r.Materials)
	}
	if r.PhoneVisible {
		t.Errorf("water bottle must not set phone_visible")
	}
}

// A drink held while also glancing down is still not cheating.
func TestBenignVarietyClear(t *testing.T) {
	for _, o := range []string{"cup", "coffee", "food", "pen", "tissue", "glass of water"} {
		r := &Report{PerFrame: frames(o, o, o, o, o, o)}
		decideVerdict(r, false, trig, false, false, true, true)
		if r.Cheating != 0 || r.ReviewRequired {
			t.Errorf("benign %q produced a flag (cheating=%d review=%v)", o, r.Cheating, r.ReviewRequired)
		}
	}
}

// ── Real cheating aids still convict ────────────────────────────────────────

func TestPaperTwoFramesCheating(t *testing.T) {
	r := &Report{PerFrame: frames("none", "paper", "paper", "none", "none", "none")}
	decideVerdict(r, false, trig, false, false, true, true)
	if r.Cheating != 1 {
		t.Fatalf("paper across 2 frames should be CHEATING, got cheating=%d", r.Cheating)
	}
	if len(r.Materials) == 0 || r.Materials[0] != "paper" {
		t.Errorf("materials should list paper, got %v", r.Materials)
	}
}

func TestPaperOneFrameReview(t *testing.T) {
	r := &Report{PerFrame: frames("none", "paper", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, false, false, true, true)
	if r.Cheating != 0 || !r.ReviewRequired {
		t.Errorf("single-frame paper should be NEEDS REVIEW, got cheating=%d review=%v", r.Cheating, r.ReviewRequired)
	}
}

func TestUnknownObjectReview(t *testing.T) {
	r := &Report{PerFrame: frames("stapler", "stapler", "stapler", "none", "none", "none")}
	decideVerdict(r, false, trig, false, false, true, true)
	if r.Cheating != 0 {
		t.Errorf("an unidentified object must not auto-convict, got cheating=%d", r.Cheating)
	}
	if !r.ReviewRequired {
		t.Errorf("an unidentified object should be NEEDS REVIEW")
	}
}

// ── Phone provenance ────────────────────────────────────────────────────────

// EVIDENCE-STRENGTH LADDER: an on-device phone hit with NO AI corroboration is
// no longer an automatic conviction. The detector can false-fire, so a detector-
// only phone (AI down / never saw it) is NEEDS REVIEW — never a silent CLEAR, but
// never a confident accusation from a single unverified sensor either.
func TestPhoneOnDeviceOnlyIsReview(t *testing.T) {
	r := &Report{PerFrame: frames("none", "none", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, true /*phoneClient*/, false /*llmPhoneSeen*/, false /*llmRan*/, false /*llmHadFrames*/)
	if r.Cheating != 0 || !r.ReviewRequired {
		t.Fatalf("detector-only phone with no AI must be NEEDS REVIEW, got cheating=%d review=%v", r.Cheating, r.ReviewRequired)
	}
	if r.PhoneVisible {
		t.Errorf("an unverified detector-only phone must not be reported as a confirmed phone violation")
	}
}

// But an on-device phone the AI ALSO saw is strong evidence → CONFIRMED.
func TestPhoneDetectorPlusAIConfirms(t *testing.T) {
	r := &Report{PerFrame: frames("phone", "none", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, true /*phoneClient*/, true /*llmPhoneSeen*/, true /*llmRan*/, true /*llmHadFrames*/)
	if r.Cheating != 1 || !r.PhoneVisible {
		t.Fatalf("detector + AI phone should be CONFIRMED, got cheating=%d phone=%v", r.Cheating, r.PhoneVisible)
	}
}

func TestPhoneBothConfirmed(t *testing.T) {
	r := &Report{PerFrame: frames("phone", "phone", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, true /*phoneClient*/, true /*llmPhoneSeen*/, true /*llmRan*/, true /*llmHadFrames*/)
	if r.Cheating != 1 || !r.PhoneVisible || !r.PhoneAIConfirmed {
		t.Fatalf("both-confirmed phone wrong: cheating=%d phone=%v ai=%v", r.Cheating, r.PhoneVisible, r.PhoneAIConfirmed)
	}
	if r.ConfidenceScore != 97 {
		t.Errorf("both-confirmed phone should be 97%%, got %d", r.ConfidenceScore)
	}
}

// ── Intruder adjudication ───────────────────────────────────────────────────

func TestFalseIntruderOverride(t *testing.T) {
	// On-device saw a 2nd face, but the LLM ran and saw NO second person → review.
	r := &Report{PerFrame: frames("none", "none", "none", "none", "none", "none")}
	decideVerdict(r, true /*clientIntruder*/, trig, false, false, true /*llmRan*/, true /*llmHadFrames*/)
	if r.IntruderConfirmed {
		t.Fatalf("intruder must NOT be confirmed when AI vision saw no second person")
	}
	if r.Cheating != 0 || !r.ReviewRequired {
		t.Errorf("unconfirmed intruder should be NEEDS REVIEW, got cheating=%d review=%v", r.Cheating, r.ReviewRequired)
	}
}

// EVIDENCE-STRENGTH LADDER: the on-device 2-face count is a flaky tripwire, so a
// detector-only intruder with NO AI confirmation is NEEDS REVIEW — not an
// automatic conviction. A second person is only CONFIRMED from ≥2 good AI frames.
func TestIntruderOnDeviceWhenLLMDownIsReview(t *testing.T) {
	r := &Report{PerFrame: frames("none", "none", "none", "none", "none", "none")}
	decideVerdict(r, true /*clientIntruder*/, trig, false, false, false /*llmRan*/, false /*llmHadFrames*/)
	if r.IntruderConfirmed || r.Cheating != 0 || !r.ReviewRequired {
		t.Errorf("detector-only intruder with no AI must be NEEDS REVIEW, got intruder=%v cheating=%d review=%v", r.IntruderConfirmed, r.Cheating, r.ReviewRequired)
	}
}

// A triggered incident the AI never analyzed must be NEEDS REVIEW — never a
// confident CLEAR that would hide it from the report (the reported bug).
func TestUnanalyzedIncidentIsReviewNotClear(t *testing.T) {
	r := &Report{PerFrame: frames("none", "none", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, false, false, false /*llmRan=false*/, false /*llmHadFrames=false*/)
	if r.Cheating != 0 {
		t.Fatalf("un-analyzed incident must not be an accusation, got cheating=%d", r.Cheating)
	}
	if !r.ReviewRequired {
		t.Errorf("un-analyzed incident must be NEEDS REVIEW (not a false CLEAR that hides it)")
	}
}

// The SAME incident, once the LLM has actually looked and seen nothing, is a
// genuine CLEAR — so we don't drown the examiner in false reviews.
func TestAnalyzedCleanIsClear(t *testing.T) {
	r := &Report{PerFrame: frames("none", "none", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, false, false, true /*llmRan=true*/, true /*llmHadFrames=true*/)
	if r.ReviewRequired || r.Cheating != 0 {
		t.Errorf("analyzed-clean should be CLEAR, got review=%v cheating=%d", r.ReviewRequired, r.Cheating)
	}
}

func TestLLMIntruder(t *testing.T) {
	pf := frames("none", "none", "none", "none", "none", "none")
	pf[1].SecondPerson = true
	pf[2].SecondPerson = true
	r := &Report{PerFrame: pf}
	decideVerdict(r, false, trig, false, false, true, true)
	if !r.IntruderConfirmed || r.Cheating != 1 {
		t.Errorf("LLM 2nd-person in 2 frames should confirm intruder, got intruder=%v cheating=%d", r.IntruderConfirmed, r.Cheating)
	}
}

// ── Regression tests from the adversarial audit ─────────────────────────────

// A cheating device listed ALONGSIDE a benign item must still convict — a benign
// word coming last must not clear the device (the "later word wins" bug).
func TestDeviceWithBenignStillCheats(t *testing.T) {
	for _, c := range []string{
		"phone and a water bottle", "water bottle and phone", "notebook and a bottle",
		"laptop next to a cup", "earbuds and a bottle", "a cup and a phone",
	} {
		if got := classifyObject(c); got != "cheat" {
			t.Errorf("classifyObject(%q) = %q, want cheat (a device must win over a co-listed benign item)", c, got)
		}
	}
	r := &Report{PerFrame: frames("none", "phone and a water bottle", "phone and a water bottle", "none", "none", "none")}
	decideVerdict(r, false, trig, false, false, true, true)
	if r.Cheating != 1 {
		t.Fatalf("a phone held next to a bottle across 2 frames must be CHEATING, got cheating=%d", r.Cheating)
	}
}

// A NEGATED object is an empty scene: "no phone" must not convict and must not review.
func TestNegatedObjectIsClear(t *testing.T) {
	for _, c := range []string{"no phone", "no object", "not holding anything", "no device", "no notes"} {
		if got := classifyObject(c); got != "none" {
			t.Errorf("classifyObject(%q) = %q, want none (a negated object is an empty scene)", c, got)
		}
	}
	r := &Report{PerFrame: frames("no phone", "no phone", "no phone", "none", "none", "none")}
	decideVerdict(r, false, trig, false, false, true, true)
	if r.Cheating != 0 || r.ReviewRequired {
		t.Errorf(`"no phone" frames must be CLEAR, got cheating=%d review=%v`, r.Cheating, r.ReviewRequired)
	}
}

// Multi-word device spellings the model emits must still classify as cheat.
func TestSpacedDeviceSpellingsCheat(t *testing.T) {
	for _, c := range []string{"ear buds", "ear piece", "wrist watch", "wristwatch", "cell phone", "smart watch", "head phones"} {
		if got := classifyObject(c); got != "cheat" {
			t.Errorf("classifyObject(%q) = %q, want cheat", c, got)
		}
	}
}

// Filler/preposition words around an empty hand must read as none, not a phantom
// unidentified object that would falsely force NEEDS REVIEW on a clean frame.
func TestFillerWordsAreClear(t *testing.T) {
	for _, c := range []string{"hand near chin", "nothing in hands", "hand resting on chin", "nothing / hand near face", "empty hand"} {
		if got := classifyObject(c); got != "none" {
			t.Errorf("classifyObject(%q) = %q, want none (empty scene)", c, got)
		}
	}
	r := &Report{PerFrame: frames("hand near chin", "nothing in hands", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, false, false, true, true)
	if r.ReviewRequired || r.Cheating != 0 {
		t.Errorf("clean/empty frames must be CLEAR, got review=%v cheating=%d", r.ReviewRequired, r.Cheating)
	}
}

// Benign plurals must never be flagged (Go de-pluralises generically).
func TestBenignPluralsClear(t *testing.T) {
	for _, c := range []string{"napkins", "tissues", "erasers", "rulers", "towels", "flasks", "tumblers", "drinks", "fruits", "cloths", "bottles", "cups"} {
		if got := classifyObject(c); got != "benign" {
			t.Errorf("classifyObject(%q) = %q, want benign", c, got)
		}
	}
}

func gazeFrames(n int) []map[string]any {
	out := make([]map[string]any, n)
	for i := 0; i < n; i++ {
		out[i] = map[string]any{"gaze": "down"}
	}
	return out
}

// buildMergedPerFrame must pin each observation to the physical frame it was
// shown, drop extra entries, and honour the model's own frame numbers — never
// misattribute an object to the wrong row or invent a phantom cheat frame.
func TestBuildMergedPerFrameAlignment(t *testing.T) {
	gaze := gazeFrames(6)
	sent := []int{0, 1, 2, 3, 5} // images sent were physical frames 0,1,2,3,5
	llm := []FrameObs{
		{Frame: 1, ObjectInHand: "none"},
		{Frame: 2, ObjectInHand: "none"},
		{Frame: 3, ObjectInHand: "none"},
		{Frame: 4, ObjectInHand: "none"},
		{Frame: 5, ObjectInHand: "book"}, // image 5 -> physical frame 5
	}
	out := buildMergedPerFrame("INC-1", nil, gaze, llm, sent)
	if len(out) != 6 {
		t.Fatalf("expected 6 rows (one per captured frame), got %d", len(out))
	}
	if out[5].ObjectInHand != "book" {
		t.Errorf("book should map to physical frame 5, got out[5]=%q", out[5].ObjectInHand)
	}
	if out[4].ObjectInHand != "none" {
		t.Errorf("physical frame 4 (never sent) should stay none, got %q", out[4].ObjectInHand)
	}
	// Provenance: every row gets a stable id, capture index, and AI-analysed flag.
	if out[5].FrameID != "INC-1-f6" || out[5].CaptureIndex != 5 || !out[5].AnalyzedAI {
		t.Errorf("frame 6 provenance wrong: id=%q idx=%d ai=%v", out[5].FrameID, out[5].CaptureIndex, out[5].AnalyzedAI)
	}
	if out[4].AnalyzedAI {
		t.Errorf("physical frame 4 was never sent to AI — AnalyzedAI must be false")
	}
	// A padded 6th entry beyond the 5 images sent must be dropped, never a new row.
	llmOver := append(append([]FrameObs{}, llm...), FrameObs{Frame: 6, ObjectInHand: "paper"})
	out2 := buildMergedPerFrame("INC-1", nil, gaze, llmOver, sent)
	for _, f := range out2 {
		if f.ObjectInHand == "paper" {
			t.Errorf("a padded 6th entry beyond the images sent must be dropped, but a 'paper' row appeared")
		}
	}
}

// A successful LLM call that returns an EMPTY per_frame must still trigger the
// false-intruder safeguard (it ran and saw no 2nd person) → NEEDS REVIEW, not a
// 90% intruder conviction from the flaky on-device 2-face count.
func TestEmptyPerFrameIntruderIsReview(t *testing.T) {
	r := &Report{PerFrame: []FrameObs{}}
	decideVerdict(r, true /*clientIntruder*/, trig, false, false, true /*llmRan*/, false /*llmHadFrames*/)
	if r.IntruderConfirmed {
		t.Fatalf("an examined-but-empty response must NOT confirm an intruder from the on-device count")
	}
	if r.Cheating != 0 || !r.ReviewRequired {
		t.Errorf("empty-per_frame + client 2-face should be NEEDS REVIEW, got cheating=%d review=%v", r.Cheating, r.ReviewRequired)
	}
}

// Replay must deterministically reproduce the stored verdict from the recorded
// ReplayInput — the core forensic guarantee ("this verdict was derived, not
// invented"). Covers cheat / clear / intruder paths and checks the trace.
func TestReplayDeterminism(t *testing.T) {
	// Phone → CONFIRMED.
	r := &Report{PerFrame: frames("phone", "phone", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, true, true, true, true)
	if r.ReplayInput == nil || r.DecisionTrace == nil {
		t.Fatalf("decideVerdict must populate ReplayInput and DecisionTrace")
	}
	if r.DecisionTrace.Rule != "confirmed" {
		t.Errorf("expected rule=confirmed, got %q", r.DecisionTrace.Rule)
	}
	if _, match := replayVerdict(*r); !match {
		t.Fatalf("phone verdict must replay-match")
	}
	// Clean → CLEAR.
	c := &Report{PerFrame: frames("none", "none", "none", "none", "none", "none")}
	decideVerdict(c, false, trig, false, false, true, true)
	if _, match := replayVerdict(*c); !match || c.DecisionTrace.Rule != "clear" {
		t.Errorf("clean verdict must replay-match with rule=clear (got %q)", c.DecisionTrace.Rule)
	}
	// Second person in ≥2 frames → INTRUDER.
	in := &Report{PerFrame: []FrameObs{{Frame: 1, SecondPerson: true}, {Frame: 2, SecondPerson: true},
		{Frame: 3}, {Frame: 4}, {Frame: 5}, {Frame: 6}}}
	decideVerdict(in, false, trig, false, false, true, true)
	if _, match := replayVerdict(*in); !match || !in.IntruderConfirmed || in.DecisionTrace.Rule != "intruder" {
		t.Errorf("intruder verdict must replay-match with rule=intruder (got %q, confirmed=%v)", in.DecisionTrace.Rule, in.IntruderConfirmed)
	}
	// A report with no ReplayInput is not reproducible.
	if _, match := replayVerdict(Report{}); match {
		t.Errorf("a report without ReplayInput must report match=false")
	}
}

// The candidate's own body, clothing, furniture and connective phrasing must
// read as an EMPTY scene — never an "unidentified object" that flags review.
func TestBodyPartsAndSceneClear(t *testing.T) {
	for _, s := range []string{
		"shoulder", "shoulders visible", "arm and shoulder", "his own shoulder",
		"hand under chin", "hand resting on desk", "hand on keyboard", "palms up",
		"looking down", "sleeve", "shirt collar", "hand behind head", "thumb",
		"elbow on table", "hair covering ear",
	} {
		if got := classifyObject(s); got != "none" {
			t.Errorf("classifyObject(%q) = %q, want none (body/scene must never flag)", s, got)
		}
	}
	// A body part in a frame row must stay CLEAR end-to-end.
	r := &Report{PerFrame: frames("none", "shoulder", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, false, false, true, true)
	if r.Cheating != 0 || r.ReviewRequired {
		t.Errorf("a shoulder frame must be CLEAR, got cheating=%d review=%v", r.Cheating, r.ReviewRequired)
	}
}

// Body-part / scene words never SHIELD a device named in the same clause.
func TestBodyPartsNeverClearADevice(t *testing.T) {
	for _, s := range []string{"phone under desk", "phone behind back", "notes on lap", "phone near shoulder"} {
		if got := classifyObject(s); got != "cheat" {
			t.Errorf("classifyObject(%q) = %q, want cheat (scene words must not clear a device)", s, got)
		}
	}
}

// Ordinary eyewear is benign; smart glasses and spaced audio-device spellings are cheat aids.
func TestEyewearAndNewDeviceSpellings(t *testing.T) {
	for _, s := range []string{"glasses", "wearing glasses", "sunglasses", "eyeglasses"} {
		if got := classifyObject(s); got != "benign" {
			t.Errorf("classifyObject(%q) = %q, want benign", s, got)
		}
	}
	for _, s := range []string{"smart glasses", "air pods", "air pods in ears", "ear pods", "cheatsheet", "chit"} {
		if got := classifyObject(s); got != "cheat" {
			t.Errorf("classifyObject(%q) = %q, want cheat", s, got)
		}
	}
}

// isPhoneObject: phone-class words only — audio devices containing "phone" don't count.
func TestIsPhoneObject(t *testing.T) {
	for s, want := range map[string]bool{
		"phone": true, "smartphone": true, "cell phone": true, "smart phone": true,
		"iphone in hand": true, "mobile": true,
		"headphone": false, "headphones": false, "earphone": false, "none": false,
		"book": false, "tablet": false, "": false,
	} {
		if got := isPhoneObject(s); got != want {
			t.Errorf("isPhoneObject(%q) = %v, want %v", s, got, want)
		}
	}
}

// EVIDENCE-STRENGTH LADDER: a phone seen by the AI in only ONE frame, with no
// on-device corroboration, is NEEDS REVIEW — one frame is not enough to convict
// (it kills the hallucinated-single-frame false accusation). A phone needs ≥2
// good frames, OR 1 good frame + detector agreement, to be CONFIRMED.
func TestSingleAIPhoneFrameIsReview(t *testing.T) {
	r := &Report{PerFrame: frames("none", "none", "smartphone", "none", "none", "none")}
	decideVerdict(r, false, trig, false /*phoneClient*/, true /*llmPhoneSeen*/, true, true)
	if r.Cheating != 0 || !r.ReviewRequired {
		t.Fatalf("a single AI-only phone frame must be NEEDS REVIEW, got cheating=%d review=%v", r.Cheating, r.ReviewRequired)
	}
	// Two good phone frames → CONFIRMED.
	r2 := &Report{PerFrame: frames("none", "smartphone", "smartphone", "none", "none", "none")}
	decideVerdict(r2, false, trig, false, true, true, true)
	if r2.Cheating != 1 || !r2.PhoneVisible {
		t.Fatalf("two good phone frames must CONFIRM, got cheating=%d phone=%v", r2.Cheating, r2.PhoneVisible)
	}
	// A 1-frame tablet (non-phone aid) is also review.
	r3 := &Report{PerFrame: frames("none", "none", "tablet", "none", "none", "none")}
	decideVerdict(r3, false, trig, false, false, true, true)
	if r3.Cheating != 0 || !r3.ReviewRequired {
		t.Errorf("a 1-frame tablet is review (needs 2 frames), got cheating=%d review=%v", r3.Cheating, r3.ReviewRequired)
	}
}

// ── Evidence-strength engine ────────────────────────────────────────────────

// qframes builds per-frame rows with an explicit quality label per frame.
func qframes(pairs ...[2]string) []FrameObs {
	out := make([]FrameObs, len(pairs))
	for i, p := range pairs {
		out[i] = FrameObs{Frame: i + 1, ObjectInHand: p[0], Quality: p[1], Gaze: "down"}
	}
	return out
}

// The four honest system-verdict states, each reachable and correctly mapped.
func TestSystemVerdictStates(t *testing.T) {
	// CONFIRMED — 2 good phone frames.
	c := &Report{PerFrame: frames("phone", "phone", "none", "none", "none", "none")}
	decideVerdict(c, false, trig, false, true, true, true)
	if c.SystemVerdict != "confirmed_violation" || c.EvidenceStrength != "strong" || c.Cheating != 1 {
		t.Errorf("2 phone frames → confirmed_violation/strong, got %q/%q cheating=%d", c.SystemVerdict, c.EvidenceStrength, c.Cheating)
	}
	// NEEDS REVIEW — 1 phone frame, no detector.
	nr := &Report{PerFrame: frames("phone", "none", "none", "none", "none", "none")}
	decideVerdict(nr, false, trig, false, true, true, true)
	if nr.SystemVerdict != "needs_review" || !nr.ReviewRequired {
		t.Errorf("1 phone frame → needs_review, got %q review=%v", nr.SystemVerdict, nr.ReviewRequired)
	}
	if nr.WhyNotClear == "" || nr.WhyNotCheating == "" {
		t.Errorf("needs_review must explain why-not-clear AND why-not-cheating; got clear=%q cheating=%q", nr.WhyNotClear, nr.WhyNotCheating)
	}
	// NO VIOLATION — clean, AI ran.
	nv := &Report{PerFrame: frames("none", "none", "none", "none", "none", "none")}
	decideVerdict(nv, false, trig, false, false, true, true)
	if nv.SystemVerdict != "no_violation" || nv.Cheating != 0 || nv.ReviewRequired {
		t.Errorf("clean analysed → no_violation, got %q cheating=%d review=%v", nv.SystemVerdict, nv.Cheating, nv.ReviewRequired)
	}
	// TECHNICAL FAILURE — no usable frames at all.
	tf := &Report{PerFrame: []FrameObs{}}
	decideVerdict(tf, false, trig, false, false, false, false)
	if tf.SystemVerdict != "technical_failure" || !tf.ReviewRequired || !tf.LLMFailed {
		t.Errorf("no usable frames → technical_failure/review/failed, got %q review=%v failed=%v", tf.SystemVerdict, tf.ReviewRequired, tf.LLMFailed)
	}
}

// A poor-quality single phone frame is WEAKER than a good one — never a
// conviction, and a phone seen only in poor frames stays review.
func TestQualityWeighting(t *testing.T) {
	// Two POOR phone frames — not "good", so not enough to confirm.
	poor := &Report{PerFrame: qframes([2]string{"phone", "poor"}, [2]string{"phone", "poor"},
		[2]string{"none", "good"}, [2]string{"none", "good"}, [2]string{"none", "good"}, [2]string{"none", "good"})}
	decideVerdict(poor, false, trig, false, true, true, true)
	if poor.SystemVerdict == "confirmed_violation" {
		t.Errorf("two POOR phone frames must not CONFIRM (need good-quality evidence), got %q", poor.SystemVerdict)
	}
	// Two GOOD phone frames — confirm.
	good := &Report{PerFrame: qframes([2]string{"phone", "good"}, [2]string{"phone", "good"},
		[2]string{"none", "good"}, [2]string{"none", "good"}, [2]string{"none", "good"}, [2]string{"none", "good"})}
	decideVerdict(good, false, trig, false, true, true, true)
	if good.SystemVerdict != "confirmed_violation" {
		t.Errorf("two GOOD phone frames must CONFIRM, got %q", good.SystemVerdict)
	}
}

// Unusable frames are excluded from the evidence tally entirely.
func TestUnusableFramesExcluded(t *testing.T) {
	a := assessEvidence(qframes([2]string{"phone", "unusable"}, [2]string{"phone", "unusable"}),
		false, trig, false, true, true, true)
	if a.UsableFrames != 0 {
		t.Errorf("unusable frames must not count as usable, got %d", a.UsableFrames)
	}
	if a.SystemVerdict != "technical_failure" {
		t.Errorf("all-unusable frames → technical_failure, got %q", a.SystemVerdict)
	}
}

// scoreFrameEvidence must fill the per-frame evidence layer for the report table.
func TestScoreFrameEvidence(t *testing.T) {
	pf := frames("phone", "stapler", "none")
	scoreFrameEvidence(pf, true /*phoneClient*/)
	if pf[0].ObjectClass != "phone" || pf[0].ObjectStatus != "present" || pf[0].FrameStatus != "violation_evidence" {
		t.Errorf("phone frame mis-scored: class=%q status=%q frame=%q", pf[0].ObjectClass, pf[0].ObjectStatus, pf[0].FrameStatus)
	}
	if !pf[0].DetectorAgree {
		t.Errorf("phone frame with phoneClient must set detector_agree")
	}
	if pf[1].ObjectStatus != "possible" && pf[1].ObjectStatus != "uncertain" {
		t.Errorf("unknown object frame should be possible/uncertain, got %q", pf[1].ObjectStatus)
	}
	if pf[2].FrameStatus != "clean" || pf[2].ObjectStatus != "not_visible" {
		t.Errorf("empty frame should be clean/not_visible, got frame=%q status=%q", pf[2].FrameStatus, pf[2].ObjectStatus)
	}
}

// The tamper-seal is deterministic and changes when any sealed field changes.
func TestTamperSeal(t *testing.T) {
	r := &Report{PerFrame: frames("phone", "phone", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, false, true, true, true)
	h1 := sealReport(r)
	if len(h1) != 64 {
		t.Fatalf("seal must be a 64-hex-char sha256, got %d chars", len(h1))
	}
	if sealReport(r) != h1 {
		t.Errorf("seal must be deterministic for the same evidence")
	}
	// Tamper with a sealed field → seal must change.
	r.Cheating = 0
	if sealReport(r) == h1 {
		t.Errorf("editing the verdict must change the tamper-seal")
	}
}

// AI failure with on-device frames present is NEEDS REVIEW; with no frames it is
// TECHNICAL FAILURE — either way it is NEVER a silent CLEAR.
func TestAIFailureNeverClear(t *testing.T) {
	withFrames := &Report{PerFrame: frames("none", "none", "none", "none", "none", "none")}
	decideVerdict(withFrames, false, trig, false, false, false /*llmRan*/, false /*llmHadFrames*/)
	if withFrames.SystemVerdict == "no_violation" {
		t.Errorf("AI failure must never be no_violation, got %q", withFrames.SystemVerdict)
	}
	if !withFrames.ReviewRequired {
		t.Errorf("AI failure with frames must be review, got review=%v", withFrames.ReviewRequired)
	}
}

// ── Hardening: weighted evidence, immutability, contradictions, timeline ────

func TestFrameWeighting(t *testing.T) {
	good := frameWeight(FrameObs{Quality: "good", AIConfidence: 90})
	poor := frameWeight(FrameObs{Quality: "poor", AIConfidence: 90})
	if good <= poor {
		t.Errorf("a good frame must weigh more than a poor one: good=%.2f poor=%.2f", good, poor)
	}
	if frameWeight(FrameObs{Quality: "unusable"}) != 0 {
		t.Errorf("an unusable frame must weigh 0")
	}
	if d := frameWeight(FrameObs{Quality: "good"}); d < 0.8 || d > 0.9 {
		t.Errorf("default (no AI confidence) weight should be ~0.85, got %.2f", d)
	}
}

// FIX 9: one blurry frame can never tip the verdict.
func TestOneBadFrameCannotConfirm(t *testing.T) {
	r := &Report{PerFrame: []FrameObs{
		{Frame: 1, ObjectInHand: "phone", Quality: "good"},
		{Frame: 2, ObjectInHand: "phone", Quality: "poor"},
		{Frame: 3, ObjectInHand: "none", Quality: "good"},
	}}
	decideVerdict(r, false, trig, false, true, true, true)
	if r.SystemVerdict == "confirmed_violation" {
		t.Errorf("one good + one poor phone frame must NOT confirm (weight below threshold), got %q", r.SystemVerdict)
	}
	if !r.ReviewRequired {
		t.Errorf("one good + one poor phone frame should be needs_review")
	}
}

// FIX 1: the raw detection primitive (no interpretation).
func TestDetectionPrimitive(t *testing.T) {
	pf := []FrameObs{
		{Frame: 1, ObjectInHand: "phone", Quality: "good"},
		{Frame: 2, ObjectInHand: "phone", Quality: "poor"},
		{Frame: 3, ObjectInHand: "stapler", Quality: "good"},
		{Frame: 4, ObjectInHand: "none", Quality: "good"},
	}
	scoreFrameEvidence(pf, false)
	want := []string{"DETECTED", "WEAK_DETECTION", "AMBIGUOUS", "NOT_DETECTED"}
	for i, w := range want {
		if pf[i].Detection != w {
			t.Errorf("frame %d detection = %q, want %q", i+1, pf[i].Detection, w)
		}
	}
}

// FIX 3: mergeFramePhones ADDS the detector signal, never overwrites the AI object.
func TestFrameTruthImmutable(t *testing.T) {
	r := &Report{PerFrame: []FrameObs{
		{Frame: 1, ObjectInHand: "none"},
		{Frame: 2, ObjectInHand: "paper"},
	}}
	mergeFramePhones(r, []bool{true, true})
	if r.PerFrame[0].ObjectInHand != "none" || r.PerFrame[1].ObjectInHand != "paper" {
		t.Errorf("mergeFramePhones must NOT overwrite the AI object, got %q %q", r.PerFrame[0].ObjectInHand, r.PerFrame[1].ObjectInHand)
	}
	if !r.PerFrame[0].DetectorPhone || !r.PerFrame[1].DetectorPhone {
		t.Errorf("mergeFramePhones must ADD the detector signal")
	}
}

// FIX 5: a detector-vs-AI disagreement is surfaced AND forces review (never confirm).
func TestContradictionSurfaced(t *testing.T) {
	a := assessEvidence([]FrameObs{
		{Frame: 1, ObjectInHand: "none", DetectorPhone: true, Quality: "good"},
		{Frame: 2, ObjectInHand: "none", DetectorPhone: true, Quality: "good"},
	}, false, trig, true /*phoneClient*/, false, true, true)
	if len(a.Contradictions) == 0 {
		t.Fatalf("a detector-phone / AI-none disagreement must be recorded as a contradiction")
	}
	if a.SystemVerdict == "confirmed_violation" {
		t.Errorf("detector-only phone (AI saw none) must NOT confirm, got %q", a.SystemVerdict)
	}
}

// FIX 6: object timeline first / peak-confidence / last.
func TestTimeline(t *testing.T) {
	tl := buildTimelines([]FrameObs{
		{Frame: 1, ObjectInHand: "none"},
		{Frame: 2, ObjectInHand: "phone", Quality: "poor", AIConfidence: 40},
		{Frame: 3, ObjectInHand: "phone", Quality: "good", AIConfidence: 95},
		{Frame: 4, ObjectInHand: "phone", Quality: "acceptable", AIConfidence: 70},
	})
	var ph *ObjectTimeline
	for i := range tl {
		if tl[i].Object == "phone" {
			ph = &tl[i]
		}
	}
	if ph == nil {
		t.Fatalf("expected a phone timeline")
	}
	if ph.FirstFrame != 2 || ph.LastFrame != 4 || ph.PeakFrame != 3 {
		t.Errorf("timeline wrong: first=%d peak=%d last=%d (want 2/3/4)", ph.FirstFrame, ph.PeakFrame, ph.LastFrame)
	}
}

// FIX 2: evidence score is a measurable number; a sensor conflict lowers it.
func TestEvidenceScoreMeasurable(t *testing.T) {
	clean := assessEvidence([]FrameObs{
		{Frame: 1, ObjectInHand: "phone", Quality: "good"},
		{Frame: 2, ObjectInHand: "phone", Quality: "good"},
	}, false, trig, false, true, true, true)
	conflicted := assessEvidence([]FrameObs{
		{Frame: 1, ObjectInHand: "phone", Quality: "good"},
		{Frame: 2, ObjectInHand: "phone", Quality: "good"},
		{Frame: 3, ObjectInHand: "none", DetectorPhone: true, Quality: "good"},
		{Frame: 4, ObjectInHand: "none", DetectorPhone: true, Quality: "good"},
	}, false, trig, true, true, true, true)
	if clean.EvidenceScore <= 0 {
		t.Errorf("a confirmed phone should have a positive evidence score, got %d", clean.EvidenceScore)
	}
	if conflicted.EvidenceScore >= clean.EvidenceScore {
		t.Errorf("conflicting detector evidence must LOWER the score: clean=%d conflicted=%d", clean.EvidenceScore, conflicted.EvidenceScore)
	}
}

// FIX 10: every verdict explains why NOT each alternative (no black box).
func TestWhyNotAllOptions(t *testing.T) {
	c := &Report{PerFrame: frames("phone", "phone", "none", "none", "none", "none")}
	decideVerdict(c, false, trig, false, true, true, true)
	if c.WhyNot["needs_review"] == "" || c.WhyNot["no_violation"] == "" {
		t.Errorf("confirmed must explain why-not-review AND why-not-clear: %+v", c.WhyNot)
	}
	r := &Report{PerFrame: frames("phone", "none", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, false, true, true, true)
	if r.WhyNot["confirmed_violation"] == "" || r.WhyNot["no_violation"] == "" {
		t.Errorf("review must explain why-not-confirmed AND why-not-clear: %+v", r.WhyNot)
	}
	n := &Report{PerFrame: frames("none", "none", "none", "none", "none", "none")}
	decideVerdict(n, false, trig, false, false, true, true)
	if n.WhyNot["confirmed_violation"] == "" || n.WhyNot["needs_review"] == "" {
		t.Errorf("clear must explain why-not-confirmed AND why-not-review: %+v", n.WhyNot)
	}
}

// ── Evidence clustering ─────────────────────────────────────────────────────

// Same-trigger incidents close in time collapse to ONE cluster; a different
// trigger or a large time gap starts a new one; the strongest verdict headlines.
func TestBuildClusters(t *testing.T) {
	pfStrong := []FrameObs{{Frame: 1, FrameStatus: "violation_evidence"}}
	incidents := []map[string]any{
		{"id": "A", "timestamp": "2026-07-03T12:00:00Z", "trigger_reason": "phone_detected_realtime", "system_verdict": "no_violation", "evidence_strength": "weak"},
		{"id": "D", "timestamp": "2026-07-03T12:00:05Z", "trigger_reason": "intruder_detected", "system_verdict": "needs_review", "evidence_strength": "moderate"},
		{"id": "B", "timestamp": "2026-07-03T12:00:10Z", "trigger_reason": "phone_detected_realtime", "system_verdict": "confirmed_violation", "evidence_strength": "strong", "per_frame": pfStrong},
		{"id": "C", "timestamp": "2026-07-03T12:01:00Z", "trigger_reason": "phone_detected_realtime", "system_verdict": "needs_review", "evidence_strength": "moderate"},
	}
	cl := buildClusters(incidents)
	if len(cl) != 3 {
		t.Fatalf("expected 3 clusters (phoneA+B, intruderD, phoneC), got %d", len(cl))
	}
	// First cluster = A+B (phones within 25s), strongest verdict (B) headlines it.
	c0 := cl[0]
	if c0["raw_count"].(int) != 2 {
		t.Errorf("cluster 1 should group 2 incidents, got %v", c0["raw_count"])
	}
	if mstr(c0, "system_verdict") != "confirmed_violation" {
		t.Errorf("cluster headline should be the STRONGEST verdict (confirmed), got %q", mstr(c0, "system_verdict"))
	}
	if c0["strong_frames"].(int) < 1 {
		t.Errorf("cluster should count the violation-evidence frame, got %v", c0["strong_frames"])
	}
	ids := c0["incident_ids"].([]string)
	if len(ids) != 2 || ids[0] != "A" || ids[1] != "B" {
		t.Errorf("cluster must preserve raw incident ids in order, got %v", ids)
	}
}

// ── Server-side PDF ─────────────────────────────────────────────────────────

func TestBuildPDFValid(t *testing.T) {
	pdf := buildPDF([][]string{{"line one", "line two"}, {"page two here"}})
	s := string(pdf)
	if !strings.HasPrefix(s, "%PDF-1.4") {
		t.Errorf("PDF must start with the header")
	}
	for _, must := range []string{"/Type /Catalog", "/Count 2", "xref", "trailer", "startxref", "%%EOF", "(line one) Tj"} {
		if !strings.Contains(s, must) {
			t.Errorf("PDF missing required token %q", must)
		}
	}
	// xref offset must point at the real "xref" keyword.
	i := strings.LastIndex(s, "startxref\n")
	if i < 0 {
		t.Fatalf("no startxref")
	}
}

func TestPdfEscape(t *testing.T) {
	if got := pdfEscape(`a(b)c\d`); got != `a\(b\)c\\d` {
		t.Errorf("pdfEscape wrong: %q", got)
	}
	if got := pdfEscape("café"); strings.ContainsRune(got, 'é') {
		t.Errorf("non-ASCII must be stripped, got %q", got)
	}
}

func TestWrapText(t *testing.T) {
	got := wrapText("hello world foo", 8)
	if len(got) != 3 {
		t.Fatalf("expected 3 wrapped lines, got %d (%v)", len(got), got)
	}
	for _, l := range got {
		if len(l) > 8 {
			t.Errorf("line exceeds width: %q", l)
		}
	}
}

// The official report renders and contains the team-lead structure.
func TestSessionReportLines(t *testing.T) {
	incidents := []map[string]any{
		{"id": "INC1", "timestamp": "2026-07-03T12:00:00Z", "trigger_reason": "phone_detected_realtime",
			"system_verdict": "confirmed_violation", "evidence_strength": "strong", "confidence_score": 97,
			"summary": "CONFIRMED cheating: a phone was clearly held.", "integrity_hash": "abc123",
			"per_frame": []FrameObs{{Frame: 1, Quality: "good", ObjectClass: "phone", ObjectStatus: "present", FrameStatus: "violation_evidence", AnalyzedAI: true}}},
	}
	clusters := buildClusters(incidents)
	signals := map[string]any{
		"total_frames": 55123.0, "duration_secs": 1123.0, "avg_fps": 44.6, "sample_secs": 6.0,
		"events": map[string]any{"total": 8.0, "phone_detected_realtime": 2.0, "intruder_detected": 1.0,
			"hand_visible_and_gaze_toward_wrist": 4.0, "covert_device_suspected": 1.0, "framing_torso_hidden": 0.0},
		"gaze":                map[string]any{"AHEAD": 40000.0, "LEFT": 3000.0, "RIGHT": 2500.0, "UP": 800.0, "DOWN": 5000.0, "NO_FACE": 200.0},
		"face":                map[string]any{"one": 54000.0, "multi": 980.0, "none": 12.0},
		"hand_present_frames": 8000.0, "phone_hint_frames": 240.0,
		"motion":   map[string]any{"avg": 12.3, "max": 88.0, "high_frames": 300.0},
		"quality":  map[string]any{"good": 50000.0, "poor": 5000.0},
		"timeline": []any{map[string]any{"sec": 0.0, "face": 1.0, "gaze": "AHEAD", "hand": false, "motion": 8.0, "phone": false}},
	}
	lines := sessionReportLines(incidents, clusters, signals, "SESSIONHASH", "2026-07-03T12:05:00Z")
	joined := strings.Join(lines, "\n")
	for _, must := range []string{"OFFICIAL SESSION REPORT", "EXECUTIVE SUMMARY", "SESSION OVERVIEW", "FRAME TIMELINE", "Total frames captured:", "EVIDENCE SUMMARY", "CLUSTER DETAILS", "PER-INCIDENT EVIDENCE", "SESSIONHASH", "INC1", "Integrity hash:"} {
		if !strings.Contains(joined, must) {
			t.Errorf("official report missing section/token %q", must)
		}
	}
}

// ── Examiner review + seal ──────────────────────────────────────────────────

// Recording an examiner decision changes the tamper-seal (the human decision is
// part of the sealed evidence chain).
func TestExaminerSealChanges(t *testing.T) {
	r := &Report{PerFrame: frames("phone", "phone", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, false, true, true, true)
	h1 := sealReport(r)
	r.ExaminerDecision = "rejected"
	r.ExaminerID = "examiner-7"
	r.ReviewedAt = "2026-07-03T12:00:00Z"
	if sealReport(r) == h1 {
		t.Errorf("recording an examiner decision must change the tamper-seal")
	}
}

func TestEnvIntDefault(t *testing.T) {
	if envIntDefault("PROCTOR_NONEXISTENT_ENV_X", 7) != 7 {
		t.Errorf("missing env must fall back to default")
	}
	os.Setenv("PROCTOR_TEST_MAX_IMG", "3")
	defer os.Unsetenv("PROCTOR_TEST_MAX_IMG")
	if envIntDefault("PROCTOR_TEST_MAX_IMG", 7) != 3 {
		t.Errorf("env override must win")
	}
	os.Setenv("PROCTOR_TEST_BAD", "-2")
	defer os.Unsetenv("PROCTOR_TEST_BAD")
	if envIntDefault("PROCTOR_TEST_BAD", 5) != 5 {
		t.Errorf("non-positive env must fall back to default")
	}
}

// End-to-end: an examiner review round-trips through report.json (fields
// persisted, seal changed, system verdict NOT overwritten) and the official PDF
// endpoint renders a valid document.
func TestReviewAndPDFEndpoints(t *testing.T) {
	id := "ZZZ-TEST-REVIEW-1"
	dir := filepath.Join(incidentsDir, id)
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	defer os.RemoveAll(dir)

	r := &Report{PerFrame: frames("phone", "phone", "none", "none", "none", "none")}
	decideVerdict(r, false, trig, false, true, true, true)
	r.IntegrityHash = sealReport(r)
	origHash, origVerdict := r.IntegrityHash, r.SystemVerdict
	b, _ := json.MarshalIndent(r, "", "  ")
	os.WriteFile(filepath.Join(dir, "report.json"), b, 0644)

	// POST /review — valid decision.
	req := httptest.NewRequest("POST", "/review?id="+id, strings.NewReader(
		`{"decision":"rejected","note":"folded paper, not a phone","examiner_id":"proctor-1"}`))
	w := httptest.NewRecorder()
	handleReview(w, req)
	if w.Code != 200 {
		t.Fatalf("review status %d: %s", w.Code, w.Body.String())
	}

	nb, _ := os.ReadFile(filepath.Join(dir, "report.json"))
	var got Report
	json.Unmarshal(nb, &got)
	if got.ExaminerDecision != "rejected" || got.ExaminerID != "proctor-1" || got.ReviewedAt == "" {
		t.Errorf("examiner fields not persisted: decision=%q id=%q at=%q", got.ExaminerDecision, got.ExaminerID, got.ReviewedAt)
	}
	if got.IntegrityHash == origHash {
		t.Errorf("seal must change after an examiner decision")
	}
	if got.SystemVerdict != origVerdict {
		t.Errorf("examiner decision must NOT overwrite the system verdict (%q -> %q)", origVerdict, got.SystemVerdict)
	}

	// Invalid decision → 400.
	wBad := httptest.NewRecorder()
	handleReview(wBad, httptest.NewRequest("POST", "/review?id="+id, strings.NewReader(`{"decision":"bogus"}`)))
	if wBad.Code != 400 {
		t.Errorf("invalid decision must be 400, got %d", wBad.Code)
	}

	// GET /session-pdf → a valid PDF that includes our incident.
	wPdf := httptest.NewRecorder()
	handleSessionPDF(wPdf, httptest.NewRequest("GET", "/session-pdf", nil))
	if ct := wPdf.Header().Get("Content-Type"); ct != "application/pdf" {
		t.Errorf("session-pdf content-type = %q, want application/pdf", ct)
	}
	if !strings.HasPrefix(wPdf.Body.String(), "%PDF") {
		t.Errorf("session-pdf did not return a PDF")
	}
}
