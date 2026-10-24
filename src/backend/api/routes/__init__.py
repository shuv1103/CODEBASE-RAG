"""Versioned API route modules.

Routers are not re-exported: they import api.core.dependencies, which builds
the API's clients at import.
"""

__all__: list[str] = []
