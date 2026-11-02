-- Run this ONCE in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run

create table if not exists public.kb_docs (
  id          uuid primary key default gen_random_uuid(),
  owner       text not null,
  key         text not null,
  name        text not null,
  digest      text not null,
  pages       int,
  words       int,
  text_pages  int,
  ocr_pages   int default 0,
  chunk_count int,
  chunk_size  int,
  overlap     int,
  embed_model text,
  emb_dim     int,
  size_bytes  bigint,
  created_at  timestamptz default now(),
  unique (owner, key)
);

-- Lock the table: only your server (service_role key) can read or write it.
alter table public.kb_docs enable row level security;

-- Private bucket for PDFs and embeddings
insert into storage.buckets (id, name, public)
values ('hh-files', 'hh-files', false)
on conflict (id) do nothing;
