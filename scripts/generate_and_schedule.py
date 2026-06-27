#!/usr/bin/env python3
"""
LinkedIn Post Agent
Pipeline: Exa (research) → Gemini (generate viral post) → LinkedIn API (post directly)

Run locally:
  python scripts/generate_and_schedule.py              # defaults to 'news' slot
  python scripts/generate_and_schedule.py --preview    # preview only, no post
  CONTENT_SLOT=educational python scripts/generate_and_schedule.py --preview

GitHub Actions triggers 4× daily at 9 AM / 1 PM / 6 PM / 10 PM IST.
"""

import os
import json
import random
import time
import requests
from datetime import datetime, timezone
from exa_py import Exa
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ── Load env (local dev; GitHub Actions injects env vars directly) ────────────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY        = os.environ.get("GEMINI_API_KEY")
GEMINI_API_KEY_2      = os.environ.get("GEMINI_API_KEY_2")
EURON_API_KEY         = os.environ.get("EURON_API_KEY")
EXA_API_KEY           = os.environ.get("EXA_API_KEY")
LINKEDIN_ACCESS_TOKEN = os.environ.get("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_PERSON_ID    = os.environ.get("LINKEDIN_PERSON_ID")

GEMINI_MODEL           = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-001"]
MAX_RETRIES            = 4
RETRY_BASE_SECONDS     = 15

# ── Platform character limits ─────────────────────────────────────────────────
PLATFORM_CHAR_LIMITS = {
    "linkedin": 3000,
    "twitter":  280,
    "x":        280,
}
PLATFORM = "linkedin"  # this pipeline posts to LinkedIn only

# ── Content slot (set by workflow; defaults to 'news') ────────────────────────
CONTENT_SLOT = os.environ.get("CONTENT_SLOT", "news")
_VALID_SLOTS = ("news", "educational", "personal", "advanced")
if CONTENT_SLOT not in _VALID_SLOTS:
    CONTENT_SLOT = "news"

# ── Load topics config ────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_script_dir, "topics.json"), "r") as f:
    _config = json.load(f)

NICHE  = _config["niche"]
PERSONA = _config["persona"]
_slot   = _config["content_slots"][CONTENT_SLOT]
SLOT_LABEL = _slot["label"]
TOPICS     = _slot["topics"]
TONES      = _slot["tones"]


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are an elite viral LinkedIn AI content writer.
Your task is to generate highly engaging LinkedIn posts ONLY about Artificial Intelligence.
Goal: maximize reach, maximize engagement (likes, comments, reposts), make people stop
scrolling in the first 2 lines, encourage comments and shares, build authority in the AI space.
Target audience: AI engineers, developers, founders, tech professionals, AI beginners who want to learn.
""".strip()

VIRAL_POST_PROMPT = """
━━━ INPUT ━━━
Persona      : {persona}
Content slot : {slot_label}
Topic        : {topic}
Tone         : {tone}

Research from the web (ground your post in this real, current data):
{research}

━━━ WRITING RULES ━━━
1. Start with a powerful hook in the first 1–2 lines.
2. The hook must create curiosity, controversy, surprise, or urgency — make people stop scrolling.
3. Use short lines (mobile-friendly). No long paragraphs.
4. Write like a human creator, not corporate marketing.
5. Mix education + storytelling + opinion.
6. Include practical insights people can learn immediately.
7. Add emotional triggers: fear of missing out, curiosity, surprise, relatability.
8. Voice: first-person always — "I", "we", "my team", "I shipped", "I broke", "I learned".
   Every post must sound like it came from someone personally in the room, not a journalist.
9. End with a CTA that drives comments (a sharp question the audience genuinely wants to answer).
10. Add 4–6 relevant hashtags at the end.

━━━ POST STRUCTURE ━━━
1. Hook (1–2 lines — stop the scroll)
2. Problem / surprising fact
3. Main insight (3–7 short punchy points or a tight story)
4. Personal opinion or takeaway
5. CTA (comment-driving question)
6. Hashtags

━━━ CONTENT RULES ━━━
✓ Max 3000 characters TOTAL (including hashtags)
✓ Short lines — single sentences or 2-line max per paragraph
✓ Specific technical terms where relevant (LLM, RAG, fine-tuning, inference, vector DB, etc.)
✓ Every claim must be specific — no vague generalities
✗ NO hype language ("game-changing", "revolutionary", "the future is here")
✗ NO generic emojis like 🚀🔥💡 — use sparingly and only if they add meaning
✗ NO bold/italic markdown — plain text only (LinkedIn renders asterisks as literals)
✗ NEVER present 3 competing ideas — pick ONE insight and go deep
✗ NEVER write from an analyst perspective — always from someone who built/shipped/broke it

━━━ OUTPUT ━━━
Return ONLY valid JSON — no prose, no markdown fences, no explanation before or after:
{{
  "post_text": "the full LinkedIn post including hashtags",
  "hook_score": <1-10 how likely this hook stops the scroll>,
  "viral_score": <1-10 overall viral potential>,
  "image_recommended": <true or false>,
  "image_type": "<infographic|meme|carousel|chart|none>",
  "image_prompt": "<detailed prompt for generating the image, or empty string if none>"
}}
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI RETRY + FALLBACK CHAIN  (key1 → key2 → Euron)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_retry_seconds(error: Exception) -> int:
    import re
    match = re.search(r"retryDelay['\"]:\s*['\"](\d+)s", str(error))
    return min(int(match.group(1)), 60) if match else RETRY_BASE_SECONDS


def _is_quota_error(error: Exception) -> bool:
    return "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error) or "quota" in str(error).lower()


def _is_retryable_server_error(error: Exception) -> bool:
    msg = str(error).lower()
    return "503" in msg or "unavailable" in msg or "high demand" in msg


def _is_daily_quota_exhausted(error: Exception) -> bool:
    s = str(error)
    return "PerDay" in s or "GenerateRequestsPerDay" in s or ("limit: 0" in s and "429" in s)


def _call_euron(prompt: str, system_instruction: str) -> str:
    if not EURON_API_KEY:
        raise RuntimeError("EURON_API_KEY not set.")
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(1, 4):
        resp = requests.post(
            "https://api.euron.one/api/v1/euri/chat/completions",
            headers={"Authorization": f"Bearer {EURON_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gemini-2.0-flash", "messages": messages},
            timeout=90,
        )
        if resp.status_code == 429:
            wait = 20 * attempt
            print(f"  [Euron] 429 rate limit, attempt {attempt}/3. Waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("Euron API failed after 3 attempts.")


def generate_text(prompt: str, system_instruction: str) -> str:
    """Call Gemini with key rotation (key1 → key2 → Euron fallback)."""
    api_keys = [k for k in [GEMINI_API_KEY, GEMINI_API_KEY_2] if k]
    models_to_try = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    last_error = None

    for key_index, api_key in enumerate(api_keys):
        client = genai.Client(api_key=api_key)
        key_label = f"key#{key_index + 1} (...{api_key[-6:]})"
        daily_exhausted = False
        print(f"  [Gemini] Trying {key_label}")

        for model_id in models_to_try:
            if daily_exhausted:
                break
            config = types.GenerateContentConfig(system_instruction=system_instruction)
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = client.models.generate_content(
                        model=model_id, contents=prompt, config=config
                    )
                    print(f"  [Gemini] Success with {model_id} on {key_label}")
                    return response.text.strip()
                except Exception as e:
                    if _is_quota_error(e) or _is_retryable_server_error(e):
                        last_error = e
                        if _is_daily_quota_exhausted(e):
                            next_key = f"key#{key_index + 2}" if key_index + 1 < len(api_keys) else "Euron fallback"
                            print(f"  [Gemini] Daily quota exhausted on {key_label}. Switching to {next_key}.")
                            daily_exhausted = True
                            break
                        wait = _parse_retry_seconds(e)
                        kind = "quota (429)" if _is_quota_error(e) else "overloaded (503)"
                        print(f"  [Gemini] {kind} on {model_id} ({key_label}), attempt {attempt}/{MAX_RETRIES}. Retrying in {wait}s...")
                        if attempt < MAX_RETRIES:
                            time.sleep(wait)
                        else:
                            print(f"  [Gemini] Retries exhausted for {model_id}, trying next model.")
                            break
                    else:
                        raise

    # All Gemini keys exhausted → try Euron
    if EURON_API_KEY:
        print("  [Euron] All Gemini keys exhausted. Falling back to Euron...")
        return _call_euron(prompt, system_instruction)

    raise last_error or RuntimeError(
        "All Gemini keys exhausted and no Euron key configured. Try again tomorrow."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Research with Exa
# ══════════════════════════════════════════════════════════════════════════════

def research_topic(topic: str, niche: str) -> str:
    """Find 5 recent high-quality articles on the topic and return a research brief."""
    print("\n[ Step 1 ] Researching topic with Exa...")

    exa = Exa(api_key=EXA_API_KEY)
    results = exa.search(
        query=f"{topic} {niche} insights trends 2025",
        type="auto",
        num_results=5,
        start_published_date="2025-01-01",
        contents={
            "text": {"max_characters": 800},
            "highlights": {"num_sentences": 3},
        },
    )

    lines = []
    for i, result in enumerate(results.results, 1):
        title      = result.title or "Untitled"
        url        = result.url
        text       = (result.text or "")[:600].strip()
        highlights = result.highlights or []

        lines.append(f"Source {i}: {title}")
        lines.append(f"URL: {url}")
        if highlights:
            lines.append(f"Key insight: {highlights[0]}")
        if text:
            lines.append(f"Context: {text[:300]}...")
        lines.append("")

    brief = "\n".join(lines)
    print(f"  Found {len(results.results)} sources.\n")
    return brief


# ══════════════════════════════════════════════════════════════════════════════
# CHARACTER LIMIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def validate_post_length(content: str, platform: str = PLATFORM) -> bool:
    """Raise ValueError if content exceeds the platform character limit."""
    limit = PLATFORM_CHAR_LIMITS.get(platform.lower(), 3000)
    if len(content) > limit:
        raise ValueError(
            f"{platform.capitalize()} posts cannot exceed {limit} characters. "
            f"Current length: {len(content)} characters."
        )
    return True


def truncate_for_platform(content: str, platform: str = PLATFORM) -> str:
    """Hard-truncate content to fit the platform limit (last-resort fallback)."""
    limit = PLATFORM_CHAR_LIMITS.get(platform.lower(), 3000)
    if len(content) <= limit:
        return content
    # Cut at the last sentence boundary within the limit
    truncated = content[:limit - 3]
    last_period = truncated.rfind(".")
    if last_period > limit // 2:
        truncated = truncated[:last_period + 1]
    else:
        truncated = truncated.rstrip() + "..."
    print(f"  [truncate] Hard-truncated to {len(truncated)} chars for {platform}.")
    return truncated


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Generate Viral Post with Gemini
# ══════════════════════════════════════════════════════════════════════════════

def generate_post(topic: str, tone: str, niche: str, persona: str, research: str) -> str:
    """Call Gemini with the viral post prompt + research brief, parse JSON response."""
    import re as _re
    import json as _json

    print("[ Step 2 ] Generating post with Gemini...")

    prompt = VIRAL_POST_PROMPT.format(
        persona=persona,
        slot_label=SLOT_LABEL,
        topic=topic,
        tone=tone,
        research=research[:2000],
    )

    raw = generate_text(prompt, SYSTEM_PROMPT)

    # Strip markdown code fences the model might wrap around JSON
    raw = raw.strip()
    raw = _re.sub(r'^```(?:json)?\s*', '', raw)
    raw = _re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    # Parse JSON; fall back to treating the whole response as post text
    hook_score = viral_score = "?"
    image_type = "none"
    image_prompt = ""
    try:
        parsed = _json.loads(raw)
        post = parsed["post_text"].strip()
        hook_score   = parsed.get("hook_score", "?")
        viral_score  = parsed.get("viral_score", "?")
        image_type   = parsed.get("image_type", "none")
        image_prompt = parsed.get("image_prompt", "")
    except (_json.JSONDecodeError, KeyError):
        print("  [warn] JSON parse failed — using raw model output as post text.")
        post = raw

    # Strip any stray markdown formatting
    post = _re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', post)
    post = _re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', post)
    post = post.strip()

    limit = PLATFORM_CHAR_LIMITS[PLATFORM]

    # If over limit, ask the model to shorten (max 2 attempts, plain text only)
    for shorten_attempt in range(2):
        if len(post) <= limit:
            break
        print(f"  Post is {len(post)} chars — asking model to shorten (attempt {shorten_attempt + 1}/2)...")
        shorten_prompt = (
            f"This LinkedIn post is {len(post)} characters, over the {limit}-character limit.\n\n"
            f"Shorten it to strictly under {limit - 50} characters while keeping the hook, "
            f"story, insights, CTA, and hashtags. Cut filler words, not ideas.\n"
            f"Plain text only — no markdown, no JSON wrapper.\n\n"
            f"Original post:\n{post}\n\n"
            f"Output ONLY the shortened post text. Nothing else."
        )
        post = generate_text(shorten_prompt, SYSTEM_PROMPT)
        post = _re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', post)
        post = _re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', post)
        post = post.strip()

    # Last-resort hard truncation
    if len(post) > limit:
        print("  AI shortening did not converge — applying hard truncation.")
        post = truncate_for_platform(post, PLATFORM)

    print(f"\n  Generated post:\n  {'─'*50}")
    for line in post.split("\n"):
        print(f"  {line}")
    print(f"  {'─'*50}")
    print(f"  Hook score : {hook_score}/10  |  Viral score : {viral_score}/10")
    print(f"  Image      : {image_type}" + (f" — {image_prompt[:80]}..." if image_prompt else ""))
    print(f"  Characters : {len(post)}/{limit}\n")

    validate_post_length(post, PLATFORM)
    return post


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Post directly to LinkedIn
# ══════════════════════════════════════════════════════════════════════════════

def post_to_linkedin(post_text: str) -> str:
    """Publish the post directly to LinkedIn via the REST Posts API."""
    print("[ Step 3 ] Posting to LinkedIn...")

    if not LINKEDIN_ACCESS_TOKEN:
        raise RuntimeError(
            "LINKEDIN_ACCESS_TOKEN is not set.\n"
            "  Run: python scripts/get_linkedin_token.py\n"
            "  Then add LINKEDIN_ACCESS_TOKEN to .env and GitHub secrets."
        )
    if not LINKEDIN_PERSON_ID:
        raise RuntimeError(
            "LINKEDIN_PERSON_ID is not set.\n"
            "  Run: python scripts/get_linkedin_token.py\n"
            "  Then add LINKEDIN_PERSON_ID to .env and GitHub secrets."
        )

    author_urn = f"urn:li:person:{LINKEDIN_PERSON_ID}"

    payload = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": post_text},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers={
                "Authorization":             f"Bearer {LINKEDIN_ACCESS_TOKEN}",
                "Content-Type":              "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
            },
            json=payload,
            timeout=15,
        )

        if response.status_code == 201:
            post_id = response.headers.get("x-restli-id", "unknown")
            print(f"  Published! LinkedIn Post ID: {post_id}\n")
            return post_id

        if response.status_code == 429:
            wait_seconds = RETRY_BASE_SECONDS * attempt
            print(f"  LinkedIn 429 rate limit, attempt {attempt}/{MAX_RETRIES}. Waiting {wait_seconds}s...")
            if attempt == MAX_RETRIES:
                raise RuntimeError("LinkedIn rate limit — too many requests. Try again tomorrow.")
            time.sleep(wait_seconds)
            continue

        if response.status_code == 401:
            raise RuntimeError(
                "LinkedIn access token is invalid or expired.\n"
                "  Run: python scripts/get_linkedin_token.py\n"
                "  Then update LINKEDIN_ACCESS_TOKEN in .env and GitHub secrets."
            )

        try:
            err = response.json()
        except ValueError:
            err = response.text
        raise RuntimeError(f"LinkedIn API error {response.status_code}: {err}")

    raise RuntimeError("LinkedIn API: exhausted retry attempts.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(preview: bool = False):
    topic = random.choice(TOPICS)
    tone  = random.choice(TONES)

    print(f"\n{'='*60}")
    print(f"  LinkedIn Post Agent — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    if preview:
        print(f"  MODE: PREVIEW (no LinkedIn posting)")
    print(f"{'='*60}")
    print(f"  Slot  : [{CONTENT_SLOT.upper()}] {SLOT_LABEL}")
    print(f"  Topic : {topic}")
    print(f"  Tone  : {tone}")
    print(f"{'='*60}\n")

    try:
        research = research_topic(topic, NICHE)
        post     = generate_post(topic, tone, NICHE, PERSONA, research)

        if preview:
            print(f"{'='*60}")
            print(f"  PREVIEW ONLY — post NOT published to LinkedIn.")
            print(f"  Run without --preview to publish it.")
            print(f"{'='*60}\n")
            return

        validate_post_length(post, PLATFORM)
        post_id = post_to_linkedin(post)

        print(f"{'='*60}")
        print(f"  Done! Post published directly to LinkedIn.")
        print(f"  LinkedIn Post ID : {post_id}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n  ERROR: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    main(preview="--preview" in sys.argv)
