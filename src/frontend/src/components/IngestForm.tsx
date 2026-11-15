import { useState, type FormEvent } from "react";
import { createRepo, getErrorMessage } from "../api";
import type { RepoStatusValue } from "../types";

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
    <form className="panel" onSubmit={handleSubmit}>
      <label className="label" htmlFor="repo-url">
        GitHub repository URL
      </label>
      <input
        id="repo-url"
        className="input mono"
        type="url"
        placeholder="https://github.com/owner/repo"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        disabled={loading}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? "repo-url-error" : undefined}
        required
        autoFocus
      />
      {error && (
        <p id="repo-url-error" className="error" role="alert">
          {error}
        </p>
      )}
      <button type="submit" className="btn btn-primary" disabled={loading}>
        {loading ? "Indexing…" : "Index repository"}
      </button>
    </form>
  );
}
