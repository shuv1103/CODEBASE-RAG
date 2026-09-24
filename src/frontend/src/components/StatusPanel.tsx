import { useEffect, useState } from "react";
import { createRepo, getErrorMessage, getRepoStatus } from "../api";
import type { RepoStatusResponse, RepoStatusValue } from "../api";

interface StatusPanelProps {
  repoId: string;
  url: string;
  onReady: () => void;
  onReset: () => void;
}

type StepState = "done" | "active" | "upcoming";

const STEPS: readonly RepoStatusValue[] = ["pending", "cloning", "chunking", "embedding"];
const POLL_MS = 2500;

export default function StatusPanel({ repoId, url, onReady, onReset }: StatusPanelProps) {
  const [record, setRecord] = useState<RepoStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function tick() {
      try {
        const data = await getRepoStatus(repoId);
        if (cancelled) return;
        setRecord(data);
        setError(null);
        if (data.status === "ready") {
          onReady();
          return;
        }
        if (data.status !== "failed") {
          timer = setTimeout(tick, POLL_MS);
        }
      } catch (err) {
        if (!cancelled) setError(getErrorMessage(err));
      }
    }

    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [repoId, attempt, onReady]);

  async function handleRetry() {
    setRetrying(true);
    setError(null);
    try {
      await createRepo(url);
      setAttempt((n) => n + 1);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setRetrying(false);
    }
  }

  const status: RepoStatusValue = record?.status ?? "pending";
  const failed = status === "failed";
  const stepIndex = STEPS.indexOf(status);
  const total = record?.chunks_total ?? null;
  const embedded = record?.chunks_embedded ?? null;
  const progressPct =
    total != null && embedded != null ? Math.min(100, Math.round((embedded / total) * 100)) : null;

  return (
    <div className="card">
      <div className="card-head">
        <span className="step-label">02 / Status</span>
        <button type="button" className="link-btn" onClick={onReset}>
          Start over
        </button>
      </div>

      <h2 className="card-title">
        {failed ? "Ingestion failed" : "Setting up your codebase"}
      </h2>
      <p className="field-label repo-url" title={url}>
        {url}
      </p>

      {!failed ? (
        <ul className="stepper">
          {STEPS.map((step, i) => {
            const state: StepState = i < stepIndex ? "done" : i === stepIndex ? "active" : "upcoming";
            return (
              <li key={step} className={`step step-${state}`}>
                <span className="step-dot" />
                <span className="step-name">{step}</span>
                {step === "embedding" && state === "active" && progressPct !== null && (
                  <span className="step-progress">
                    {embedded}/{total}
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="error-text">{record?.error || "Something went wrong during ingestion."}</p>
      )}

      {!failed && status === "embedding" && progressPct !== null && (
        <div className="progress-bar">
          <span style={{ width: `${progressPct}%` }} />
        </div>
      )}

      {error && <p className="error-text">{error}</p>}

      {failed && (
        <button type="button" className="btn-primary" onClick={handleRetry} disabled={retrying}>
          {retrying ? "Retrying…" : "Try again"}
        </button>
      )}
    </div>
  );
}
