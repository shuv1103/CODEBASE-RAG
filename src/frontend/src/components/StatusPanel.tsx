import { useEffect, useState } from "react";
import { ApiError, createRepo, getErrorMessage, getRepoStatus } from "../api";
import type { RepoStatusResponse, RepoStatusValue } from "../types";

interface StatusPanelProps {
  repoId: string;
  url: string;
  name: string;
  onReady: () => void;
}

const STEPS: RepoStatusValue[] = ["pending", "cloning", "chunking", "embedding"];
const POLL_MS = 2500;

export default function StatusPanel({ repoId, url, name, onReady }: StatusPanelProps) {
  const [record, setRecord] = useState<RepoStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);
  const [stopped, setStopped] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let done = false;
    let inFlight = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    // A hidden tab schedules nothing; becoming visible checks at once.
    function schedule() {
      if (!document.hidden) timer = setTimeout(tick, POLL_MS);
    }

    async function tick() {
      timer = undefined;
      inFlight = true;
      try {
        const data = await getRepoStatus(repoId);
        if (cancelled) return;
        setRecord(data);
        setError(null);
        // Ready once means its old index still serves chat during a re-ingest.
        if (data.status === "ready" || data.ready_at) {
          done = true;
          onReady();
          return;
        }
        if (data.status === "failed") {
          done = true;
          return;
        }
        schedule();
      } catch (err) {
        if (cancelled) return;
        setError(getErrorMessage(err));
        // A 4xx other than 429 (unknown or expired repo) won't fix itself.
        if (err instanceof ApiError && err.status >= 400 && err.status < 500 && err.status !== 429) {
          done = true;
          setStopped(true);
          return;
        }
        // Keep polling: one failed check (network blip, 503) must not freeze the panel.
        schedule();
      } finally {
        inFlight = false;
      }
    }

    function handleVisibility() {
      if (document.hidden) {
        clearTimeout(timer);
        timer = undefined;
      } else if (!timer && !inFlight && !done) {
        tick();
      }
    }

    document.addEventListener("visibilitychange", handleVisibility);
    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [repoId, attempt, onReady]);

  async function handleRetry() {
    setRetrying(true);
    setError(null);
    setStopped(false);
    try {
      await createRepo(url);
      setRecord(null);
      setAttempt((n) => n + 1);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setRetrying(false);
    }
  }

  const status: RepoStatusValue = record?.status ?? "pending";
  const failed = status === "failed" || stopped;
  const stepIndex = STEPS.indexOf(status);
  const total = record?.chunks_total ?? null;
  const embedded = record?.chunks_embedded ?? null;
  const counting = status === "embedding" && total != null && embedded != null && total > 0;
  const ratio = counting ? Math.min(1, embedded / total) : 0;

  if (failed) {
    return (
      <div className="panel">
        <h2 className="panel-title">Couldn’t index {name}</h2>
        <p className="error-box" role="alert">
          {(!stopped && record?.error) || "Indexing stopped before it finished."}
        </p>
        {error && <p className="error">{error}</p>}
        <button type="button" className="btn btn-primary" onClick={handleRetry} disabled={retrying}>
          {retrying ? "Retrying…" : "Try again"}
        </button>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2 className="panel-title">Indexing {name}</h2>

      <div className="reading" aria-live="polite">
        {counting ? (
          <span className="reading-value mono">
            {embedded} / {total}
          </span>
        ) : (
          <span className="reading-value">{STEPS[stepIndex] ?? "pending"}</span>
        )}
        <span className="reading-caption">
          {counting ? "chunks indexed" : `Step ${Math.max(stepIndex, 0) + 1} of ${STEPS.length}`}
        </span>
        <div
          className="meter"
          role="progressbar"
          aria-label="Chunks indexed"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(ratio * 100)}
        >
          <span style={{ transform: `scaleX(${ratio})` }} />
        </div>
      </div>

      <ol className="trail">
        {STEPS.map((step, i) => {
          const state = i < stepIndex ? "done" : i === stepIndex ? "active" : "todo";
          return (
            <li key={step} className={`trail-step is-${state}`} aria-current={state === "active" ? "step" : undefined}>
              <span className="trail-mark" aria-hidden="true" />
              <span>{step}</span>
              <span className="trail-state mono">{state === "done" ? "done" : state === "active" ? "running" : ""}</span>
            </li>
          );
        })}
      </ol>

      {error && (
        <p className="notice" role="status">
          {error}. Retrying.
        </p>
      )}
    </div>
  );
}
