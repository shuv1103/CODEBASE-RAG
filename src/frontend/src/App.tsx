import { useCallback, useState } from "react";
import IngestForm from "./components/IngestForm";
import StatusPanel from "./components/StatusPanel";
import ChatPanel from "./components/ChatPanel";
import type { RepoStatusValue } from "./types";

type Stage = "ingest" | "status" | "chat";

interface ActiveRepo {
  repoId: string;
  url: string;
  name: string;
}

const STAGES: { id: Stage; label: string }[] = [
  { id: "ingest", label: "Repository" },
  { id: "status", label: "Indexing" },
  { id: "chat", label: "Chat" },
];

// "https://github.com/pallets/flask.git" -> "pallets/flask"
function repoName(url: string): string {
  return url
    .replace(/^https?:\/\/(www\.)?github\.com\//i, "")
    .replace(/\.git$/, "")
    .replace(/\/+$/, "");
}

export default function App() {
  const [stage, setStage] = useState<Stage>("ingest");
  const [repo, setRepo] = useState<ActiveRepo | null>(null);

  function handleIngestStarted(repoId: string, url: string, initialStatus: RepoStatusValue) {
    setRepo({ repoId, url, name: repoName(url) });
    setStage(initialStatus === "ready" ? "chat" : "status");
  }

  const handleReady = useCallback(() => setStage("chat"), []);

  function handleReset() {
    setRepo(null);
    setStage("ingest");
  }

  const current = STAGES.findIndex((s) => s.id === stage);

  return (
    <div className={`app app-${stage}`}>
      <header className="topbar">
        <span className="wordmark">Codebase RAG</span>
        <ol className="progress-nav" aria-label="Progress">
          {STAGES.map((s, i) => (
            <li
              key={s.id}
              className={i < current ? "is-done" : i === current ? "is-current" : undefined}
              aria-current={i === current ? "step" : undefined}
            >
              {s.label}
            </li>
          ))}
        </ol>
        {repo && (
          <div className="topbar-repo">
            <span className="repo-chip" title={repo.url}>
              {repo.name}
            </span>
            <button type="button" className="btn btn-quiet" onClick={handleReset}>
              {stage === "chat" ? "New repository" : "Start over"}
            </button>
          </div>
        )}
      </header>

      {stage === "chat" && repo ? (
        <ChatPanel repoId={repo.repoId} name={repo.name} />
      ) : (
        <main className="split">
          {stage === "ingest" && <IngestForm onStarted={handleIngestStarted} />}
          {stage === "status" && repo && (
            <StatusPanel repoId={repo.repoId} url={repo.url} name={repo.name} onReady={handleReady} />
          )}
        </main>
      )}
    </div>
  );
}
