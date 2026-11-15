"""Pure ASGI middlewares: request ids, body size limit, generic 500s."""

import logging
import re
import uuid

from fastapi import HTTPException
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from utils.logging_config import request_id_var

logger = logging.getLogger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"
# A client-sent id lands in every log line, so only plain tokens are kept.
_VALID_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class RequestIdMiddleware:
    """ASGI middleware that tags a request's logs with an id.

    Reuses a safe client X-Request-ID (1-64 chars of [A-Za-z0-9._-]), else
    generates a uuid4 hex, and echoes it as the X-Request-ID response header.

    Args:
        app: The ASGI app to wrap.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Run the request with its id in log context and on the response."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        sent = (
            dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        )
        request_id = sent if _VALID_ID.fullmatch(sent) else uuid.uuid4().hex

        async def send_with_id(message: Message) -> None:
            """Add the X-Request-ID header to the response start message."""
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[_REQUEST_ID_HEADER] = request_id
            await send(message)

        token = request_id_var.set(request_id)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


_BODY_TOO_LARGE_DETAIL = "Request body too large"


class BodySizeLimitMiddleware:
    """ASGI middleware that rejects request bodies over max_bytes with 413.

    Checks Content-Length up front and counts streamed bytes too, since a
    client can omit or understate the header.

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
        """Pass an HTTP request through, enforcing the body size limit."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (
            not declared.isdigit() or int(declared) > self.max_bytes
        ):
            await JSONResponse(
                status_code=413, content={"detail": _BODY_TOO_LARGE_DETAIL}
            )(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            """Return the next ASGI message, counting body bytes.

            Raises:
                HTTPException: 413 once the body passes max_bytes.
            """
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI re-raises HTTPException from body parsing;
                    # any other error would become a 400.
                    raise HTTPException(
                        status_code=413, detail=_BODY_TOO_LARGE_DETAIL
                    )
            return message

        await self.app(scope, limited_receive, send)


class UnhandledErrorMiddleware:
    """ASGI middleware that turns any unhandled error into a generic 500.

    A FastAPI handler for Exception runs outside every middleware, so its
    500s would lack CORS headers (the browser hides them) and a request id.
    Added innermost, this keeps both; details are logged, never returned.

    Args:
        app: The ASGI app to wrap.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Run the request; answer an escaped error with a generic 500.

        Raises:
            Exception: An error raised after the response already started.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def track_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, track_send)
        except Exception:
            if response_started:
                raise
            logger.exception("Unhandled error in %s", scope["path"])
            response = JSONResponse(
                status_code=500, content={"detail": "Internal server error"}
            )
            await response(scope, receive, send)
