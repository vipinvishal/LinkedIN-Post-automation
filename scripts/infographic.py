"""
Infographic generator for LinkedIn posts.

Pipeline:
  generate_content()  → Gemini → 5-box content dict
  render_infographic()  → Playwright → PNG
  upload_to_linkedin()  → LinkedIn media API → asset URN
"""

import os
import re
import json
import pathlib
import requests

INFOGRAPHIC_HANDLE = os.environ.get("INFOGRAPHIC_HANDLE", "@VipinAIHub")

_CONTENT_PROMPT = """
Generate content for a hand-drawn sketchbook flowchart infographic about an AI topic.

LAYOUT — 5 nodes connected by arrows (like the LinkedIn 30-day plan style):
- box1 (top-left)    : Starting point / context / input
- box2 (top-right)   : Output / result / what you get
- box4 (CENTER)      : Core process / main concept (the hub everything connects through)
- box3 (bottom-left) : Supporting input / resource / feed-in element
- box5 (bottom-right): Measurement / tracking / outcome

Topic: {topic}
Research context: {research}

STRICT RULES:
- All box labels : 1–2 words, ALL CAPS (e.g. "STRATEGY", "RESULTS", "ANALYTICS")
- All box points : exactly 3 items, max 4 words each — keep short and punchy
- title_line1 : first 2–3 words of the topic, ALL CAPS
- title_line2 : remaining words, ALL CAPS (split so each line is similar length)

Return ONLY valid JSON — no markdown, no explanation:
{{
  "title_line1": "...",
  "title_line2": "...",
  "box1": {{"label": "...", "points": ["...", "...", "..."]}},
  "box2": {{"label": "...", "points": ["...", "...", "..."]}},
  "box3": {{"label": "...", "points": ["...", "...", "..."]}},
  "box4": {{"label": "...", "points": ["...", "...", "..."]}},
  "box5": {{"label": "...", "points": ["...", "...", "..."]}}
}}
""".strip()

_SYSTEM = "You generate structured JSON content for visual infographics. Return only valid JSON, no extra text."


def generate_content(topic: str, research: str, generate_text_fn) -> dict:
    """Call the LLM to produce the 5-node infographic content dict."""
    prompt = _CONTENT_PROMPT.format(topic=topic, research=research[:1200])

    for attempt in range(2):
        raw = generate_text_fn(prompt, _SYSTEM)
        raw = raw.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()
        try:
            data = json.loads(raw)
            required = ["title_line1", "title_line2", "box1", "box2", "box3", "box4", "box5"]
            if all(k in data for k in required):
                data["handle"] = INFOGRAPHIC_HANDLE
                return data
        except (json.JSONDecodeError, KeyError):
            pass

    raise RuntimeError("Failed to generate valid infographic JSON after 2 attempts.")


def render_infographic(content: dict, out_path: str) -> str:
    """Render the content dict to a PNG using Playwright. Returns PNG path."""
    import sys
    root = pathlib.Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from renderer.render import render
    return render(content, out_path)


def upload_to_linkedin(png_path: str, access_token: str, person_id: str) -> str:
    """
    Upload PNG to LinkedIn via the media upload API.
    Returns the asset URN to embed in the ugcPost.
    """
    headers_json = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type":  "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }

    # Step 1 — register upload
    reg = requests.post(
        "https://api.linkedin.com/v2/assets?action=registerUpload",
        headers=headers_json,
        json={
            "registerUploadRequest": {
                "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                "owner":   f"urn:li:person:{person_id}",
                "serviceRelationships": [{
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent",
                }],
            }
        },
        timeout=20,
    )
    reg.raise_for_status()
    val = reg.json()["value"]
    upload_url = val["uploadMechanism"][
        "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
    ]["uploadUrl"]
    asset_urn = val["asset"]

    # Step 2 — upload image bytes
    with open(png_path, "rb") as f:
        img_bytes = f.read()

    put = requests.put(
        upload_url,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "image/png"},
        data=img_bytes,
        timeout=60,
    )
    put.raise_for_status()

    print(f"  [infographic] Uploaded → {asset_urn}")
    return asset_urn
