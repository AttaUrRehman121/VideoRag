"""
Vercel entrypoint.

Just re-exports the existing ASGI app from config.asgi so
all Django wiring stays in the main project.
"""
from config.asgi import application as app


