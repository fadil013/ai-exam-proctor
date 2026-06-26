-- Proctor AI — incidents table
-- Run once in the Supabase SQL Editor:
-- https://supabase.com/dashboard/project/hvzsxwbgkxdevjfzvnkg/sql/new

create table if not exists public.incidents (
  id                         text primary key,
  created_at                 timestamptz default now(),
  timestamp                  text,
  cheating                   int  default 0,
  confidence_score           int,
  cheating_likelihood        text,
  cheating_material_detected boolean default false,
  materials                  jsonb default '[]'::jsonb,
  suspicious_behaviors       jsonb default '[]'::jsonb,
  gaze_analysis              text,
  visual_evidence            text,
  recommendation             text,
  summary                    text,
  gaze_data                  jsonb,
  collage                    text          -- base64 data URL of the blurred collage
);

create index if not exists incidents_created_at_idx on public.incidents (created_at desc);
