"""
Infographic generator for LinkedIn posts.

Pipeline:
  generate_content()  → Gemini → 5-step content dict
  render_infographic()  → Playwright → PNG
  upload_to_linkedin()  → LinkedIn media API → asset URN
"""

import os
import re
import json
import pathlib
import requests

_CONTENT_PROMPT = """
Generate content for a dark-mode, high-contrast LinkedIn infographic breaking an AI/tech topic into
a clear 5-step sequential mental model (STEP 1 → STEP 2 → STEP 3 → STEP 4 → STEP 5, read top to bottom).
This is for an individual engineer's personal learning post — technical, precise, no business framing.
Write for someone scrolling fast on mobile: the headline must stop the scroll, and each step must be
understandable in under 2 seconds.

Topic: {topic}

The LinkedIn post this infographic will accompany (the infographic MUST illustrate the SAME narrative —
same problem, same solution/technique, same specific claims. Do not introduce a different angle, a
different solution, or new facts not present in this post):
{post_text}

STRICT RULES:
- box1..box5 : five sequential steps that walk through the SAME story as the post above (its problem →
  its solution → its key points), step 1 is the simplest entry point, step 5 is the payoff / end state.
  If the post lists specific features/bullets for its solution, box2-5 should reflect those specific
  points rather than inventing new ones.
- All box labels : 1–2 words, ALL CAPS (e.g. "RETRIEVAL", "LATENCY", "GPU MEMORY")
- All box points : exactly 3 items, max 3 words each — short tag-like phrases, plain text (no markdown)
- title_line1 : a short, punchy, curiosity-driving hook phrase (3-5 words) matching the post's hook,
  NOT the raw topic name. Should read like a scroll-stopping headline, e.g. "WHY YOUR RAG" not
  "Retrieval Augmented Generation".
- title_line2 : the payoff / rest of the hook (3-5 words), ALL CAPS, completes the headline from line1.
- hook : one short punchy sentence (max 14 words) — the same core takeaway as the post's closing thought.
  Written like a terminal code comment. Plain text, not ALL CAPS.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "title_line1": "...",
  "title_line2": "...",
  "hook": "...",
  "box1": {{"label": "...", "points": ["...", "...", "..."]}},
  "box2": {{"label": "...", "points": ["...", "...", "..."]}},
  "box3": {{"label": "...", "points": ["...", "...", "..."]}},
  "box4": {{"label": "...", "points": ["...", "...", "..."]}},
  "box5": {{"label": "...", "points": ["...", "...", "..."]}}
}}
""".strip()

_PROCESS_CONTENT_PROMPT = """
Generate content for a light-theme, detailed "how it works" LinkedIn infographic that walks through a
step-by-step AI/tech mechanism (like a diagram explaining how a process runs end to end). This is for an
individual engineer's personal learning post — technical, precise, no business framing.

Topic: {topic}

The LinkedIn post this infographic will accompany (the infographic MUST illustrate the SAME narrative —
same mechanism, same specific claims. Do not introduce facts not present in this post):
{post_text}

This infographic has THREE sections. All of it must walk through the SAME mechanism, at increasing detail:

1. "stages" — exactly 3 items representing INPUT → PROCESS → OUTPUT for this mechanism. Each has:
   - "label": 2-3 words, Title Case (e.g. "You Write Code", "Model Predicts", "Token Sampled")
   - "snippet": a short fake code/terminal/data line (max 22 characters) that visually represents that
     stage — e.g. for input: 'print("hello")', for process: '01001 → 01110', for output: '> Hello!'

2. "steps" — exactly 4 numbered cards walking through the mechanism in more detail than "stages" (step 1
   is the simplest entry point, step 4 is the payoff/end state). Each has:
   - "label": 2-4 words, Title Case
   - "points": exactly 3 short tag-like phrases, max 3 words each, plain text

3. Two flow summaries of the SAME mechanism at different granularities:
   - "flow_a_items": 5-6 short words/phrases (1-2 words each) — the detailed/granular pipeline
   - "flow_b_items": 4-5 short words/phrases (1-2 words each) — the simplified big-picture version

Also:
- title_line1 : a short, punchy, curiosity-driving hook phrase (3-5 words) matching the post's hook,
  Title Case, NOT the raw topic name.
- title_line2 : the payoff / rest of the hook (3-5 words), Title Case, completes line1.
- tagline : one short descriptive line (under 10 words), e.g. "From typing code to seeing the output."
- hook : one short punchy sentence (max 20 words) — the same core takeaway as the post's closing thought.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "title_line1": "...", "title_line2": "...", "tagline": "...", "hook": "...",
  "stages": [{{"label": "...", "snippet": "..."}}, {{"label": "...", "snippet": "..."}}, {{"label": "...", "snippet": "..."}}],
  "steps": [
    {{"label": "...", "points": ["...", "...", "..."]}},
    {{"label": "...", "points": ["...", "...", "..."]}},
    {{"label": "...", "points": ["...", "...", "..."]}},
    {{"label": "...", "points": ["...", "...", "..."]}}
  ],
  "flow_a_items": ["...", "...", "...", "...", "..."],
  "flow_b_items": ["...", "...", "...", "..."]
}}
""".strip()

_SYSTEM = "You generate structured JSON content for visual infographics. Return only valid JSON, no extra text."


def _clean_text(s: str) -> str:
    """Strip stray markdown/comment markers the model sometimes leaks into 'plain text' fields."""
    s = s.strip()
    s = re.sub(r'^(#+|//+)\s*', '', s)
    s = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', s)
    s = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', s)
    s = s.replace('`', '')
    return s.strip()


def _clean_content(data: dict) -> dict:
    data["title_line1"] = _clean_text(data["title_line1"])
    data["title_line2"] = _clean_text(data["title_line2"])
    data["hook"] = _clean_text(data["hook"])
    for key in ["box1", "box2", "box3", "box4", "box5"]:
        data[key]["label"] = _clean_text(data[key]["label"])
        data[key]["points"] = [_clean_text(p) for p in data[key]["points"]]
    return data


def generate_content(topic: str, post_text: str, generate_text_fn) -> dict:
    """Call the LLM to produce the 5-node infographic content dict, grounded in the actual post text."""
    prompt = _CONTENT_PROMPT.format(topic=topic, post_text=post_text[:2500])

    for attempt in range(2):
        raw = generate_text_fn(prompt, _SYSTEM)
        raw = raw.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()
        try:
            data = json.loads(raw)
            required = ["title_line1", "title_line2", "hook", "box1", "box2", "box3", "box4", "box5"]
            if all(k in data for k in required):
                return _clean_content(data)
        except (json.JSONDecodeError, KeyError):
            pass

    raise RuntimeError("Failed to generate valid infographic JSON after 2 attempts.")


def _clean_process_content(data: dict) -> dict:
    data["title_line1"] = _clean_text(data["title_line1"])
    data["title_line2"] = _clean_text(data["title_line2"])
    data["tagline"] = _clean_text(data["tagline"])
    data["hook"] = _clean_text(data["hook"])
    for stage in data["stages"]:
        stage["label"] = _clean_text(stage["label"])
        stage["snippet"] = _clean_text(stage["snippet"])
    for step in data["steps"]:
        step["label"] = _clean_text(step["label"])
        step["points"] = [_clean_text(p) for p in step["points"]]
    data["flow_a_items"] = [_clean_text(i) for i in data["flow_a_items"]]
    data["flow_b_items"] = [_clean_text(i) for i in data["flow_b_items"]]
    return data


def generate_process_content(topic: str, post_text: str, generate_text_fn) -> dict:
    """Call the LLM to produce the 'how it works' process-infographic content dict."""
    prompt = _PROCESS_CONTENT_PROMPT.format(topic=topic, post_text=post_text[:2500])

    for attempt in range(2):
        raw = generate_text_fn(prompt, _SYSTEM)
        raw = raw.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()
        try:
            data = json.loads(raw)
            required = ["title_line1", "title_line2", "tagline", "hook",
                        "stages", "steps", "flow_a_items", "flow_b_items"]
            if all(k in data for k in required) and len(data["stages"]) == 3 and len(data["steps"]) == 4:
                return _clean_process_content(data)
        except (json.JSONDecodeError, KeyError):
            pass

    raise RuntimeError("Failed to generate valid process-infographic JSON after 2 attempts.")


def render_infographic(content: dict, out_path: str, template: str = "infographic.html.j2") -> str:
    """Render the content dict to a PNG using Playwright. Returns PNG path."""
    import sys
    root = pathlib.Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from renderer.render import render
    return render(content, out_path, template=template)


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
