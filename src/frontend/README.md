# Frontend

Minimal React + TypeScript UI for the Codebase RAG API (the project's
`src/api/` FastAPI app — see `CLAUDE.md` §15 for the design brief and §14
for the pipeline this drives). Three states in one card: paste a GitHub URL
→ watch ingestion status → chat. Lives at `src/frontend/` — a separate Vite
app, not part of the Python package.

## Run

```bash
cd src/frontend
npm install
cp .env.example .env   # point VITE_API_BASE_URL at the running FastAPI backend
npm run dev
```

Backend must be running separately (`uvicorn api.main:app` from the
project's `src/`, or `docker-compose up`) and reachable at
`VITE_API_BASE_URL`.

`npm run build` type-checks (`tsc --noEmit`, strict mode) before bundling —
a type error fails the build. Run `npm run typecheck` on its own for a
quicker check while iterating.

## Structure

```
src/
  api.ts                    # every backend call + response types — the only file that knows the HTTP contract
  App.tsx                    # stage state machine: ingest -> status -> chat
  vite-env.d.ts                # Vite client types + import.meta.env typing
  components/
    IngestForm.tsx             # 01 — submit a GitHub URL
    StatusPanel.tsx             # 02 — polls GET /api/repos/{id} until ready/failed
    ChatPanel.tsx                 # 03 — Q/A against POST /api/repos/{id}/chat
  styles/
    index.css                   # tokens + page layout
    components.css               # card/button/input/stepper/chat styles
```

`tsconfig.json` is `strict: true` plus `noUnusedLocals`,
`noUnusedParameters`, `noImplicitReturns`, `noFallthroughCasesInSwitch`, and
`noUncheckedIndexedAccess` — one config file, no project-references split,
since this app is small enough not to need one.

No chat history is persisted (in-memory only, per session) and no
`assistant-ui` dependency was pulled in — the chat surface here is a plain
request/response Q&A (no streaming, no multi-thread state), so a ~90-line
component covers it without the extra runtime.
