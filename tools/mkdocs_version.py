"""MkDocs hook: put the package version (src/tssa/__init__.py) in the page footer and on the home page.

The version is read from the source text, so building the docs never imports tssa. Pages can use the
placeholder ``{{ tssa_version }}``.
"""

import re
from pathlib import Path

_INIT = Path(__file__).resolve().parents[1] / "src" / "tssa" / "__init__.py"


def _version() -> str:
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', _INIT.read_text(encoding="utf-8"), re.M)
    if not m:
        raise RuntimeError(f"no __version__ in {_INIT}")
    return m.group(1)


def on_config(config):
    config["copyright"] = f"tssa {_version()}"
    return config


def on_page_markdown(markdown, page, config, files):
    return markdown.replace("{{ tssa_version }}", _version())
