from __future__ import annotations

import re
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "ui"


def test_index_references_existing_assets():
    html = (UI / "index.html").read_text(encoding="utf-8")
    local = re.findall(r'(?:src|href)="((?!https?:)[^"]+)"', html)
    assert local
    for path in local:
        assert (UI / path).is_file(), path


def test_module_imports_resolve():
    for module in (UI / "js").glob("*.js"):
        for target in re.findall(r'from "(\./[^"]+)"', module.read_text(encoding="utf-8")):
            assert (module.parent / target).is_file(), f"{module.name} imports {target}"


def test_every_element_id_used_by_the_app_exists():
    html = (UI / "index.html").read_text(encoding="utf-8")
    script = (UI / "js" / "app.js").read_text(encoding="utf-8")
    for element_id in re.findall(r'getElementById\("([^"]+)"\)', script):
        assert f'id="{element_id}"' in html, element_id
