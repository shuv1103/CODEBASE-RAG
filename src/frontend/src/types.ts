// Mirrors the FastAPI request/response schemas.
export type RepoStatusValue =
  | "pending"
  | "cloning"
  | "chunking"
  | "embedding"
  | "ready"
  | "failed";

export type CacheHit = "exact" | "semantic" | null;

export interface CreateRepoResponse {
  repo_id: string;
  status: RepoStatusValue;
}

export interface RepoStatusResponse {
  repo_id: string;
  url: string;
  status: RepoStatusValue;
  chunks_total: number | null;
  chunks_embedded: number | null;
  error: string | null;
  created_at: string;
  ready_at: string | null;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  cache_hit: CacheHit;
}
