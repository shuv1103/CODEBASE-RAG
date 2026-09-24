import { useState, type FormEvent } from "react";
import { createRepo, getErrorMessage } from "../api";
import type { RepoStatusValue } from "../api";

interface IngestFormProps {
  onStarted: (repoId: string, url: string, initialStatus: RepoStatusValue) => void;
}

export default function IngestForm({ onStarted }: IngestFormProps) {
  const [url, setUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError(null);
    try {
      const res = await createRepo(trimmed);
      onStarted(res.repo_id, trimmed, res.status);
    } catch (err) {
      setError(getErrorMessage(err));
      setLoading(false);
    }
  }

  return (
    <form className="card" onSubmit={handleSubmit}>
      <div className="card-head">
        <span className="step-label">01 / Repository</span>
        <span className="chevron">&rsaquo;</span>
      </div>

      <h2 className="card-title">Point it at your codebase</h2>

      <label className="field-label" htmlFor="repo-url">
        Github URL
      </label>
      <div className="input-wrap">
        <svg className="input-icon" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
          <path d="M8 0a8 8 0 0 0-2.53 15.59c.4.07.55-.17.55-.38v-1.5c-2.22.48-2.69-1.07-2.69-1.07-.36-.92-.88-1.16-.88-1.16-.72-.49.05-.48.05-.48.8.06 1.22.82 1.22.82.71 1.21 1.87.86 2.33.66.07-.52.28-.86.5-1.06-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.58.82-2.14-.08-.2-.36-1.02.08-2.13 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.03 2.2-.82 2.2-.82.44 1.11.16 1.93.08 2.13.51.56.82 1.27.82 2.14 0 3.07-1.87 3.75-3.65 3.94.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 8 0Z" />
        </svg>
        <input
          id="repo-url"
          type="url"
          placeholder="https://github.com/owner/repository"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={loading}
          required
        />
      </div>

      {error && <p className="error-text">{error}</p>}

      <button type="submit" className="btn-primary" disabled={loading}>
        {loading ? "Ingesting…" : "Ingest repository"}
        <span className="btn-arrow" aria-hidden="true">↗</span>
      </button>

      <p className="helper-text">
        <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor" aria-hidden="true">
          <path d="M8 0a3 3 0 0 0-3 3v2H4a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1h-1V3a3 3 0 0 0-3-3ZM6.5 3a1.5 1.5 0 0 1 3 0v2h-3V3Z" />
        </svg>
        Your repository stays scoped to this workspace
      </p>
    </form>
  );
}
