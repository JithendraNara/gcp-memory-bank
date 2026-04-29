"""pytest config — keep pytest from trying to collect plugin source files.

The plugin uses relative imports (``from .client import ...``) which only
work when imported as a package. The test bootstrap copies the directory
under an underscored name; pytest must not import the source location.
"""

collect_ignore_glob = [
    "__init__.py", "config.py", "client.py", "topics.py", "sessions.py",
    "ingestion.py", "retrieval.py", "synthesize.py", "tools.py", "cli.py",
    "observability.py",
]
