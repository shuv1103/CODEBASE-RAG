"""ASGI middleware that tags each request with an X-Request-ID."""

import re
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from utils.logging_config import request_id_var

_HEADER = "X-Request-ID"
# A client-sent id lands in every log line, so only plain tokens are kept.
_VALID_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class RequestIdMiddleware:
    """Set request_id_var for the request and echo it as X-Request-ID.

    A valid client-sent X-Request-ID is reused, so a caller can correlate
    its own logs; anything else gets a fresh uuid4 hex.

    Args:
        app: The ASGI app to wrap.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Run the request with its id in context and on the response.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        sent = (
            dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        )
        request_id = sent if _VALID_ID.fullmatch(sent) else uuid.uuid4().hex

        async def send_with_id(message: Message) -> None:
            """Add the X-Request-ID header to the response start."""
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[_HEADER] = request_id
            await send(message)

        token = request_id_var.set(request_id)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)
