#!/usr/bin/env python3
"""Render an infographic PNG from a content dict.

Usage (CLI):
    python renderer/render.py                  # renders sample content
    python renderer/render.py out.png          # custom output path

Usage (module):
    from renderer.render import render
    png_path = render(content_dict, "renderer/output/infographic.png")
"""
import sys
import json
import pathlib
import tempfile

from jinja2 import Environment, FileSystemLoader

ROOT      = pathlib.Path(__file__).parent
TEMPLATE  = "infographic.html.j2"
SAMPLE    = ROOT / "data" / "sample_content.json"
OUTPUT    = ROOT / "output" / "infographic.png"


def _build_html(content: dict) -> str:
    env  = Environment(loader=FileSystemLoader(str(ROOT / "templates")))
    tmpl = env.get_template(TEMPLATE)
    return tmpl.render(**content)


def _draw(page) -> None:
    """Trigger JS arrow drawing + autofit after fonts are confirmed loaded."""
    page.evaluate("window.drawInfographic && window.drawInfographic()")


def render(content: dict, out_path: str) -> str:
    """Render content dict → PNG. Returns out_path string."""
    from playwright.sync_api import sync_playwright

    html = _build_html(content)

    # Write HTML to temp file so file:// URL works for relative asset loading
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(html)
        tmp_html = f.name

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": 900, "height": 920},
                device_scale_factor=2,   # 2× = crisp 1800×1840 output
            )
            page.goto(f"file://{tmp_html}")
            page.wait_for_load_state("networkidle")   # waits for Google Fonts
            page.wait_for_timeout(800)                # extra buffer for font apply
            _draw(page)                               # draw arrows from real positions
            page.wait_for_timeout(150)                # let SVG paint
            el = page.query_selector(".page")
            el.screenshot(path=str(out_path))
            browser.close()
    finally:
        import os
        os.unlink(tmp_html)

    print(f"  [render] → {out_path}")
    return str(out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(OUTPUT)
    if SAMPLE.exists():
        sample = json.loads(SAMPLE.read_text())
    else:
        sample = {
            "title_line1": "Why AI Agents",
            "title_line2": "Fail in Production",
            "box1": {"label": "THE DEMO",    "points": ["static inputs",   "happy paths only", "controlled data"]},
            "box2": {"label": "PRODUCTION",  "points": ["dynamic inputs",  "real edge cases",  "concurrent load"]},
            "box3": {"label": "ROOT CAUSE",  "points": ["no validation",   "brittle logic",    "missing fallbacks"]},
            "box4": {"label": "THE GAP",     "points": ["demo ≠ prod",     "false confidence"]},
            "box5": {"label": "THE FIX",     "points": ["guardrails",      "observability",    "retry logic"]},
            "quote": "Production success = boring engineering details",
            "handle": "@VipinAIHub",
        }
    render(sample, out)
    print(f"Saved to {out}")
