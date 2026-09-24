import { useEffect, useRef, useState, type FormEvent } from "react";
import { getErrorMessage, sendChatMessage } from "../api";

interface ChatPanelProps {
  repoId: string;
  url: string;
  onReset: () => void;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  cached?: boolean;
}

function ExpandIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {expanded ? (
        <>
          <path d="M6 1.5v4.5H1.5" />
          <path d="M0.75 0.75 6 6" />
          <path d="M8 12.5V8h4.5" />
          <path d="M13.25 13.25 8 8" />
        </>
      ) : (
        <>
          <path d="M8.5 1h4.5v4.5" />
          <path d="M13 1 8 6" />
          <path d="M5.5 13H1V8.5" />
          <path d="M1 13l5-5" />
        </>
      )}
    </svg>
  );
}

export default function ChatPanel({ repoId, url, onReset }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const listRef = useRef<HTMLDivElement | null>(null);

  function scrollToBottom() {
    requestAnimationFrame(() => {
      const el = listRef.current;
      el?.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    });
  }

  // Lock the page behind the panel and let Escape collapse it.
  useEffect(() => {
    if (!expanded) return;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") setExpanded(false);
    }
    window.addEventListener("keydown", handleKey);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKey);
    };
  }, [expanded]);

  // Keep the newest message in view when the panel changes size.
  useEffect(scrollToBottom, [expanded]);

  async function handleSend(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setSending(true);
    setError(null);
    scrollToBottom();

    try {
      const res = await sendChatMessage(repoId, text, sessionId);
      setSessionId(res.session_id);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.answer, cached: res.cached },
      ]);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSending(false);
      scrollToBottom();
    }
  }

  return (
    <>
      {expanded && (
        <div className="chat-backdrop" onClick={() => setExpanded(false)} aria-hidden="true" />
      )}
      <div className={`card chat-card${expanded ? " chat-card-expanded" : ""}`}>
        <div className="card-head">
          <span className="step-label">03 / Chat</span>
          <div className="card-head-actions">
            <button type="button" className="link-btn" onClick={onReset}>
              New repository
            </button>
            <button
              type="button"
              className="icon-btn"
              onClick={() => setExpanded((v) => !v)}
              aria-expanded={expanded}
              aria-label={expanded ? "Collapse chat" : "Expand chat to full screen"}
              title={expanded ? "Collapse (Esc)" : "Expand to full screen"}
            >
              <ExpandIcon expanded={expanded} />
            </button>
          </div>
        </div>
        <p className="field-label repo-url" title={url}>
          {url}
        </p>

        <div className="chat-list" ref={listRef}>
          {messages.length === 0 && (
            <p className="chat-empty">Ask a question about this codebase to get started.</p>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`bubble bubble-${m.role}`}>
              {m.cached && <span className="bubble-cached-badge">cached</span>}
              {m.content}
            </div>
          ))}
          {sending && (
            <div className="bubble bubble-assistant bubble-pending">Searching the codebase…</div>
          )}
        </div>

        {error && <p className="error-text">{error}</p>}

        <form className="chat-input-row" onSubmit={handleSend}>
          <input
            type="text"
            placeholder="What does this repo do?"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={sending}
          />
          <button type="submit" className="btn-primary btn-compact" disabled={sending || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </>
  );
}
