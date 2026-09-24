// Single point of contact with the FastAPI backend (see CLAUDE.md §15.3 for
// the verified request/response contract this mirrors).
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export type RepoStatusValue =
  | "pending"
  | "cloning"
  | "chunking"
  | "embedding"
  | "ready"
  | "failed";

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
  cached: boolean;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong.";
}

function extractErrorMessage(data: unknown, status: number): string {
  if (data && typeof data === "object" && "detail" in data) {
    const detail = (data as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) =>
          item && typeof item === "object" && "msg" in item
            ? String((item as { msg: unknown }).msg)
            : String(item)
        )
        .join(", ");
    }
  }
  return `Request failed (${status})`;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data: unknown = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new ApiError(extractErrorMessage(data, res.status), res.status);
  }
  return data as T;
}

export function createRepo(url: string): Promise<CreateRepoResponse> {
  return request<CreateRepoResponse>("/api/repos", {
    method: "POST",
    body: JSON.stringify({ url }),
  });
}

export function getRepoStatus(repoId: string): Promise<RepoStatusResponse> {
  return request<RepoStatusResponse>(`/api/repos/${repoId}`);
}

export function sendChatMessage(
  repoId: string,
  message: string,
  sessionId: string | null
): Promise<ChatResponse> {
  return request<ChatResponse>(`/api/repos/${repoId}/chat`, {
    method: "POST",
    body: JSON.stringify({ message, session_id: sessionId }),
  });
}
