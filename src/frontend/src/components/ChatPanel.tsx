import { memo, useEffect, useRef, useState, type FormEvent } from "react";
import { getErrorMessage, sendChatMessage } from "../api";
import type { CacheHit } from "../types";

interface ChatPanelProps {
  repoId: string;
  name: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  cacheHit?: CacheHit;
}

// Memoized so typing in the composer does not re-render every message.
const MessageList = memo(function MessageList({
  messages,
  sending,
  name,
}: {
  messages: ChatMessage[];
  sending: boolean;
  name: string;
}) {
  return (
    <>
      {messages.length === 0 && <p className="chat-empty">Ask anything about {name}.</p>}
      {messages.map((m, i) => (
        <div key={i} className={`msg msg-${m.role}`}>
          {m.cacheHit && <span className="tag">{m.cacheHit === "exact" ? "Exact cache" : "Semantic cache"}</span>}
          {m.content}
        </div>
      ))}
      {sending && <div className="msg msg-assistant msg-pending">Searching the code…</div>}
    </>
  );
});

export default function ChatPanel({ repoId, name }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  // Keep the newest message in view after every send and reply.
  useEffect(() => {
    const el = listRef.current;
    el?.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  async function handleSend(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setSending(true);
    setError(null);

    try {
      const res = await sendChatMessage(repoId, text, sessionId);
      setSessionId(res.session_id);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.answer, cacheHit: res.cache_hit },
      ]);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSending(false);
    }
  }

  return (
    <main className="chat">
      <div className="chat-log" ref={listRef} role="log" aria-live="polite">
        <div className="chat-column">
          <MessageList messages={messages} sending={sending} name={name} />
        </div>
      </div>

      <div className="chat-dock">
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <form className="composer" onSubmit={handleSend}>
          <input
            className="composer-input"
            type="text"
            aria-label="Your question"
            placeholder="Ask a question about the code"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={sending}
            autoFocus
          />
          <button type="submit" className="btn btn-primary" disabled={sending || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </main>
  );
}
