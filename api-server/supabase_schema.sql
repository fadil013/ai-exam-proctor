-- Proctor AI — incidents table
-- Run once in the Supabase SQL Editor:
-- https://supabase.com/dashboard/project/hvzsxwbgkxdevjfzvnkg/sql/new
--
-- This matches EXACTLY the columns the Go server writes (IncidentRecord) and reads
-- (fetchSupabase SELECT). The server upserts on the id primary key
-- (Prefer: resolution=merge-duplicates, on_conflict=id), so an incident is written
-- once as a provisional verdict and then UPDATED in place with the AI-enriched
-- verdict — the dashboard always shows the latest state, never a stale row.

create table if not exists public.incidents (
  id                         text primary key,
  created_at                 timestamptz default now(),
  timestamp                  text,
  cheating                   int     default 0,
  review_required            boolean default false,
  confidence_score           int,
  cheating_likelihood        text,
  cheating_material_detected boolean default false,
  phone_visible              boolean default false,
  phone_detected             boolean default false,
  phone_ai_confirmed         boolean default false,
  face_count                 int     default 1,
  intruder_detected          boolean default false,
  intruder_confirmed         boolean default false,
  trigger_reason             text,
  materials                  jsonb   default '[]'::jsonb,
  suspicious_behaviors       jsonb   default '[]'::jsonb,
  gaze_analysis              text,
  visual_evidence            text,
  recommendation             text,
  summary                    text,
  gaze_data                  jsonb,
  per_frame                  jsonb   default '[]'::jsonb,
  llm_pending                boolean default false,
  llm_failed                 boolean default false,
  system_verdict             text,         -- confirmed_violation | needs_review | no_violation | technical_failure
  evidence_strength          text,         -- strong | moderate | weak | inconclusive
  why_not_clear              text,
  why_not_cheating           text,
  integrity_hash             text,         -- sha256 tamper-seal over the frozen evidence
  collage                    text          -- base64 data URL of the blurred collage
);

-- Bring an EXISTING table up to date without dropping it (safe to re-run).
alter table public.incidents add column if not exists review_required    boolean default false;
alter table public.incidents add column if not exists phone_visible      boolean default false;
alter table public.incidents add column if not exists phone_detected     boolean default false;
alter table public.incidents add column if not exists phone_ai_confirmed boolean default false;
alter table public.incidents add column if not exists face_count         int     default 1;
alter table public.incidents add column if not exists intruder_detected  boolean default false;
alter table public.incidents add column if not exists intruder_confirmed boolean default false;
alter table public.incidents add column if not exists trigger_reason     text;
alter table public.incidents add column if not exists per_frame          jsonb   default '[]'::jsonb;
alter table public.incidents add column if not exists llm_pending        boolean default false;
alter table public.incidents add column if not exists llm_failed         boolean default false;
alter table public.incidents add column if not exists system_verdict     text;
alter table public.incidents add column if not exists evidence_strength  text;
alter table public.incidents add column if not exists why_not_clear      text;
alter table public.incidents add column if not exists why_not_cheating   text;
alter table public.incidents add column if not exists integrity_hash     text;

create index if not exists incidents_created_at_idx on public.incidents (created_at desc);
