"""ASGI middleware that rejects oversized request bodies with HTTP 413."""

from fastapi import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_DETAIL = "Request body too large"


class BodySizeLimitMiddleware:
    """Reject request bodies larger than max_bytes with HTTP 413.

    Schema length limits only apply after the whole body is buffered, so
    this checks Content-Length up front and also counts the streamed bytes,
    since a client can omit or lie about Content-Length.

    Args:
        app: The ASGI app to wrap.
        max_bytes: Largest accepted request body, in bytes.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Pass the request through, enforcing the body size limit.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (
            not declared.isdigit() or int(declared) > self.max_bytes
        ):
            await JSONResponse(status_code=413, content={"detail": _DETAIL})(
                scope, receive, send
            )
            return

        received = 0

        async def limited_receive() -> Message:
            """Receive the next message; raise 413 once max_bytes is passed."""
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI re-raises HTTPException from body parsing;
                    # any other error would become a 400.
                    raise HTTPException(status_code=413, detail=_DETAIL)
            return message

        await self.app(scope, limited_receive, send)
