import { useCallback, useState } from "react";
import IngestForm from "./components/IngestForm";
import StatusPanel from "./components/StatusPanel";
import ChatPanel from "./components/ChatPanel";
import type { RepoStatusValue } from "./api";

type Stage = "ingest" | "status" | "chat";

interface ActiveRepo {
  repoId: string;
  url: string;
}

export default function App() {
  const [stage, setStage] = useState<Stage>("ingest");
  const [repo, setRepo] = useState<ActiveRepo | null>(null);

  const handleIngestStarted = useCallback(
    (repoId: string, url: string, initialStatus: RepoStatusValue) => {
      setRepo({ repoId, url });
      setStage(initialStatus === "ready" ? "chat" : "status");
    },
    []
  );

  const handleReady = useCallback(() => setStage("chat"), []);

  const handleReset = useCallback(() => {
    setRepo(null);
    setStage("ingest");
  }, []);

  return (
    <div className="page">
      <section className="hero">
        <p className="eyebrow">
          <span className="eyebrow-dash" />
          Read the repository
        </p>
        <h1>
          Ask better
          <br />
          <span className="accent">code questions.</span>
        </h1>
        <p className="hero-copy">
          Ingest one GitHub repository. Then talk to an assistant that stays
          close to its source. No broad web answers. No invented context.
        </p>
      </section>

      <section className="panel-wrap">
        {stage === "ingest" && <IngestForm onStarted={handleIngestStarted} />}
        {stage === "status" && repo && (
          <StatusPanel
            repoId={repo.repoId}
            url={repo.url}
            onReady={handleReady}
            onReset={handleReset}
          />
        )}
        {stage === "chat" && repo && (
          <ChatPanel repoId={repo.repoId} url={repo.url} onReset={handleReset} />
        )}
      </section>
    </div>
  );
}
