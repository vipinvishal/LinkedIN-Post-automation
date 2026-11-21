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
from hook_matrix import select_hook_formula, POWER_PHRASES

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

GEMINI_MODEL           = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-flash-latest"]
MAX_RETRIES            = 4
GEMINI_MAX_ATTEMPTS    = 2    # per model; move on to the next model quickly when overloaded
GEMINI_TIMEOUT_MS      = 60000
RETRY_BASE_SECONDS     = 15

# ── Platform character limits ─────────────────────────────────────────────────
PLATFORM_CHAR_LIMITS = {
    "linkedin": 3000,
    "twitter":  280,
    "x":        280,
}
PLATFORM = "linkedin"  # this pipeline posts to LinkedIn only
TARGET_POST_LENGTH = 1300  # soft target for a short, scannable post — well under the 3000 hard cap

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
You are ghostwriting LinkedIn posts for ONE individual engineer who studies AI/ML deeply every day and
shares what they learn in public. This person is an individual contributor learning in public — NOT a
founder, NOT a CEO, NOT a tech lead, NOT a manager, and does not speak for any company or team.
Your task is to generate highly engaging, educational LinkedIn posts about how AI actually works —
model architecture, GPUs and chips, distributed training, inference serving, RAG, evaluation, and the
real systems and engineering running behind the scenes of AI.
Goal: teach people something real about the technology, maximize engagement (likes, comments, reposts),
make people stop scrolling in the first 2 lines, encourage comments and shares, build authority through
technical depth — not hype.
Target audience: AI/ML engineers, developers, tech professionals, and AI beginners who want to genuinely
learn how the technology works.
VOICE — read this carefully: you are a hands-on ENGINEER, not a senior architect or expert lecturing from
a whiteboard. Never explain a mechanism the way a textbook or a senior architect briefing junior engineers
would ("Think about it:", "Here's how it works:", confident declarative statements about what "the model
does"). Instead, ground everything in specific things YOU did — I ran, I tested, I read, I debugged, I
noticed, I got confused by, I re-checked. If you wouldn't say a line out loud to a peer over coffee, cut it.
STRICTLY OUT OF SCOPE: business news, funding rounds, valuations, stock moves, market share, company
rivalry/drama, layoffs, IPOs, career/business-advice angles, or anything implying the writer leads a team,
runs a company, or manages an org's AI strategy. This is one engineer's personal learning journey — never
a company update, product announcement, or leadership narrative. If a topic drifts toward business or
leadership framing, redirect it to the underlying technology instead.
""".strip()

STYLE_1_PROMPT = """
━━━ INPUT ━━━
Persona      : {persona}
Content slot : {slot_label}
Topic        : {topic}
Tone         : {tone}

Research from the web (ground your post in this real, current data):
{research}

━━━ FORMAT: PROBLEM → SOLUTION ━━━
Follow this exact structure. Each numbered part is 1-2 short sentences MAX, not a paragraph:
1. Opening hook — this line (and the 1-2 that follow it) has to work standalone, before LinkedIn's
   "see more" cutoff (~210 characters). Use this exact hook formula: {hook_name}.
   {hook_skeleton}
   NEVER open with a command or imperative ("Stop doing X", "Here's how to Y") — that style
   measurably kills engagement. It should read like an observation, not an order.
2. Emphasize the problem — one line on why it's worse than it looks, or what it costs people who
   ignore it. Don't repeat point 1 in different words.
3. Name the specific technical solution directly — the actual technique, architecture, tool, or
   concept from the research/topic that solves the problem. Skip the anticipation-building, go
   straight to it. It must genuinely fit this AI/tech topic — never a generic or unrelated product name.
4. Highlight exactly 3 concrete features/capabilities of that solution that prove it solves the
   problem from steps 1-2. One line each. Specific and technical, not vague marketing language.
5. One closing line that reframes how readers should think about this problem going forward.
6. One short, natural line encouraging readers to follow your profile for more content like this
   (not salesy).

━━━ WRITING RULES ━━━
- Voice: first-person, one individual ENGINEER sharing what they ran into and figured out — NEVER "we",
  "my team", "our roadmap", "at my company", or anything implying you lead a team or company. You are also
  NOT a senior architect lecturing junior engineers — avoid "Think about it:", "Here's how it works:", or
  confidently declaring what "the model does" like a textbook. Ground the explanation in what YOU actually
  did (I ran, I tested, I debugged, I noticed) rather than presenting mechanisms from a position of authority.
- No hype language ("game-changing", "revolutionary", "the future is here").
- No bold/italic markdown, and never use "*" as a bullet marker anywhere — LinkedIn renders asterisks
  as literal characters, not formatting. Use "→" or a plain "-" for any bullet/list line instead.
- Short, mobile-friendly lines — 1-2 sentences per paragraph, never more. No throat-clearing, no
  restating the same point twice in different words.
- No business/funding/company content — stay on the technology itself.
- No hashtags.
- TARGET LENGTH: 900-1300 characters total. This is a post someone reads end-to-end in one glance while
  scrolling, not an essay — every sentence must earn its place. 3000 characters is the hard ceiling, not
  the goal.
- NEVER open with a question (measurably the single worst opener). NEVER open with "Here's what/how/why"
  or a "Stop X, start Y" line. NEVER bridge a reveal with "The result?" or "Plot twist:" — just state it.
  NEVER announce candor ("Let me be honest", "Confession:", "unpopular opinion") without a specific,
  dated fact right behind it — the fact alone is what earns the reaction, the announcement is a tell.
- At most ONE contrast frame in the whole post ("it's not X, it's Y" or similar) — most posts should
  use zero. At most ONE rule-of-three, and it must be three concrete numbers or named things, never
  three interchangeable adjectives.
- The closing question (if used) goes only at the very end, never earlier, and must name the specific
  thing this post is about — never a generic "Thoughts?", "What do you think?", or "Agree or disagree?".
- Optional power phrases you may use naturally, filled in with real specifics, if one genuinely fits
  (never force one in, never use more than one per post): {power_phrases}

━━━ OUTPUT ━━━
Return ONLY valid JSON — no prose, no markdown fences, no explanation before or after:
{{
  "post_text": "the full LinkedIn post (no hashtags)",
  "hook_score": <1-10 how likely this hook stops the scroll>,
  "viral_score": <1-10 overall viral potential>,
  "image_recommended": <true or false>,
  "image_type": "<infographic|meme|carousel|chart|none>",
  "image_prompt": "<detailed prompt for generating the image, or empty string if none>",
  "first_comment": "<a short, punchy comment (1-2 sentences) that YOU, the author, would drop as the
    first reply right after posting this — a quick hot take on the topic above, or a direct, specific
    invite for people to share their own experience/opinion in the comments. Plain text, no hashtags,
    no links, no markdown, no generic 'thoughts?' filler — it must clearly connect to what this
    particular post said.>"
}}
""".strip()

STYLE_2_PROMPT = """
━━━ INPUT ━━━
Persona      : {persona}
Content slot : {slot_label}
Topic        : {topic}
Tone         : {tone}

Research from the web (ground your post in this real, current data):
{research}

━━━ FORMAT: SCENARIO → RISK → SOLUTION ━━━
Follow this exact structure. Each numbered part is 1-2 short sentences MAX except the bullets in step 5:
1. Opening hook — this line has to work standalone, before LinkedIn's "see more" cutoff (~49
   characters). Start with an imaginary scenario — a short, vivid "Imagine..." or "Picture this..."
   moment that puts the reader inside a real situation tied to this topic (a 2am pager alert, a demo
   breaking in front of a customer, a model silently failing in production — whatever genuinely fits
   the topic). If the research has a strong concrete number, work it into this opening beat instead —
   stat-driven hooks measurably outperform pure scenario-setting. Either way this must read as a
   specific, vivid moment, never a command or imperative ("Stop doing X", "Here's how to Y").
2. One line on why this matters at real scale — why it's not just a toy-project problem.
3. One line naming the specific risk — what actually goes wrong technically. Don't repeat point 2.
4. Introduce the solution — name the actual technique, architecture, tool, or concept from the
   research/topic that addresses this risk. It must genuinely fit this AI/tech topic — never a generic
   or unrelated product name.
5. Cover the solution in exactly 3 short bullet points — specific, technical, no filler, one line each.
   Prefix each bullet with "→ " (an arrow, not an asterisk or dash) — LinkedIn renders plain text only,
   and a literal "*" shows up as a stray character instead of a bullet.
6. Conclude with a line that invites the audience into the comments — a sharp, specific question.

━━━ WRITING RULES ━━━
- Voice: first-person, one individual ENGINEER, not a senior architect or expert lecturing from authority
  — NEVER "we", "my team", "our roadmap", "at my company", or anything implying you lead a team or
  company. Outside the scenario itself, ground the risk/solution explanation in what YOU noticed or
  looked into, not confident textbook-style declarations. The imaginary scenario can be told in second
  person ("Imagine you...") or about a hypothetical team — that's a narrative device, not a claim about
  your own company.
- No hype language ("game-changing", "revolutionary" and similar).
- No bold/italic markdown, and never use "*" as a bullet marker anywhere — LinkedIn renders asterisks
  as literal characters, not formatting. Use "→" or a plain "-" for any bullet/list line instead.
- Short, mobile-friendly lines — 1-2 sentences per paragraph, never more. No throat-clearing, no
  restating the same point twice in different words.
- No business/funding/company content — stay on the technology itself.
- No hashtags.
- TARGET LENGTH: 900-1300 characters total. This is a post someone reads end-to-end in one glance while
  scrolling, not an essay — every sentence must earn its place. 3000 characters is the hard ceiling, not
  the goal.
- NEVER open with a question (measurably the single worst opener). NEVER open with "Here's what/how/why"
  or a "Stop X, start Y" line. NEVER bridge a reveal with "The result?" or "Plot twist:" — just state it.
  NEVER announce candor ("Let me be honest", "Confession:", "unpopular opinion") without a specific,
  dated fact right behind it — the fact alone is what earns the reaction, the announcement is a tell.
- At most ONE contrast frame in the whole post ("it's not X, it's Y" or similar) — most posts should
  use zero. At most ONE rule-of-three, and it must be three concrete numbers or named things, never
  three interchangeable adjectives.
- The closing question goes only at the very end, never earlier, and must name the specific thing this
  post is about — never a generic "Thoughts?", "What do you think?", or "Agree or disagree?".
- Optional power phrases you may use naturally, filled in with real specifics, if one genuinely fits
  (never force one in, never use more than one per post): {power_phrases}

━━━ OUTPUT ━━━
Return ONLY valid JSON — no prose, no markdown fences, no explanation before or after:
{{
  "post_text": "the full LinkedIn post (no hashtags)",
  "hook_score": <1-10 how likely this hook stops the scroll>,
  "viral_score": <1-10 overall viral potential>,
  "image_recommended": <true or false>,
  "image_type": "<infographic|meme|carousel|chart|none>",
  "image_prompt": "<detailed prompt for generating the image, or empty string if none>",
  "first_comment": "<a short, punchy comment (1-2 sentences) that YOU, the author, would drop as the
    first reply right after posting this — a quick hot take on the topic above, or a direct, specific
    invite for people to share their own experience/opinion in the comments. Plain text, no hashtags,
    no links, no markdown, no generic 'thoughts?' filler — it must clearly connect to what this
    particular post said.>"
}}
""".strip()

# Deterministic alternation (not random) so both styles get an even, predictable rotation to compare —
# day-of-year + fixed slot position means every slot cycles through both styles day to day.
_SLOT_ORDER = ["news", "educational", "personal", "advanced"]


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
    return any(x in msg for x in ("503", "504", "unavailable", "high demand", "deadline", "timed out", "timeout", "disconnected"))


def _is_daily_quota_exhausted(error: Exception) -> bool:
    s = str(error)
    return "PerDay" in s or "GenerateRequestsPerDay" in s or ("limit: 0" in s and "429" in s)


def _is_model_not_found(error: Exception) -> bool:
    s = str(error)
    return "404" in s and ("NOT_FOUND" in s or "not found" in s.lower())


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
            json={"model": "gemini-3.6-flash", "messages": messages},
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
        # google-genai 3.x defaults to API v1, which rejects system instructions
        # ("Developer instruction is not enabled for api version v1"). Pin v1beta.
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta", timeout=GEMINI_TIMEOUT_MS),
        )
        key_label = f"key#{key_index + 1} (...{api_key[-6:]})"
        daily_exhausted = False
        print(f"  [Gemini] Trying {key_label}")

        for model_id in models_to_try:
            if daily_exhausted:
                break
            config = types.GenerateContentConfig(system_instruction=system_instruction)
            for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
                try:
                    response = client.models.generate_content(
                        model=model_id, contents=prompt, config=config
                    )
                    print(f"  [Gemini] Success with {model_id} on {key_label}")
                    return response.text.strip()
                except Exception as e:
                    if _is_model_not_found(e):
                        last_error = e
                        print(f"  [Gemini] {model_id} not found/retired on {key_label}. Trying next model.")
                        break
                    if _is_quota_error(e) or _is_retryable_server_error(e):
                        last_error = e
                        if _is_daily_quota_exhausted(e):
                            next_key = f"key#{key_index + 2}" if key_index + 1 < len(api_keys) else "Euron fallback"
                            print(f"  [Gemini] Daily quota exhausted on {key_label}. Switching to {next_key}.")
                            daily_exhausted = True
                            break
                        wait = _parse_retry_seconds(e)
                        kind = "quota (429)" if _is_quota_error(e) else "overloaded (503)"
                        print(f"  [Gemini] {kind} on {model_id} ({key_label}), attempt {attempt}/{GEMINI_MAX_ATTEMPTS}. Retrying in {wait}s...")
                        if attempt < GEMINI_MAX_ATTEMPTS:
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

def generate_post(topic: str, tone: str, niche: str, persona: str, research: str) -> tuple[str, str]:
    """Call Gemini with the viral post prompt + research brief, parse JSON response.
    Returns (post_text, first_comment)."""
    import re as _re
    import json as _json

    slot_index = _SLOT_ORDER.index(CONTENT_SLOT) if CONTENT_SLOT in _SLOT_ORDER else 0
    day_of_year = datetime.now(timezone.utc).timetuple().tm_yday
    use_style_2 = (day_of_year + slot_index) % 2 == 1
    template = STYLE_2_PROMPT if use_style_2 else STYLE_1_PROMPT

    # STYLE_2 keeps its own built-in scenario cold-open (a hook formula in its own right).
    # STYLE_1's opening line is driven by the hook matrix, rotating deterministically by
    # day-of-year + slot so every formula gets exercised evenly over time.
    hook = select_hook_formula(CONTENT_SLOT, day_of_year, slot_index)
    power_phrases = "; ".join(POWER_PHRASES)

    print(f"[ Step 2 ] Generating post with Gemini... (style: {'2 (scenario/risk/solution)' if use_style_2 else '1 (problem/solution)'})")
    if not use_style_2:
        print(f"  Hook formula: {hook['id']} — {hook['name']}")

    prompt = template.format(
        persona=persona,
        slot_label=SLOT_LABEL,
        topic=topic,
        tone=tone,
        research=research[:2000],
        hook_name=hook["name"],
        hook_skeleton=hook["skeleton"],
        power_phrases=power_phrases,
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
    first_comment = f"Curious how others are handling this — what's your take on {topic.lower()}?"
    try:
        parsed = _json.loads(raw)
        post = parsed["post_text"].strip()
        hook_score    = parsed.get("hook_score", "?")
        viral_score   = parsed.get("viral_score", "?")
        image_type    = parsed.get("image_type", "none")
        image_prompt  = parsed.get("image_prompt", "")
        first_comment = parsed.get("first_comment", "").strip() or first_comment
    except (_json.JSONDecodeError, KeyError):
        # Model likely emitted an unescaped quote inside post_text, breaking strict JSON.
        # Recover just the post_text field via regex instead of dumping raw JSON as the post.
        match = _re.search(r'"post_text"\s*:\s*"(.*)"\s*,\s*"hook_score"', raw, _re.DOTALL)
        if match:
            print("  [warn] JSON parse failed — recovered post_text via regex.")
            post = match.group(1)
            post = post.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        else:
            print("  [warn] JSON parse failed and post_text not recoverable — using raw model output as post text.")
            post = raw

    # Strip any stray markdown formatting
    post = _re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', post)
    post = _re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', post)
    post = _re.sub(r'^\s*\*\s+', '→ ', post, flags=_re.MULTILINE)  # stray "* " bullet → arrow
    post = post.strip()

    first_comment = _re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', first_comment)
    first_comment = _re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', first_comment).strip()

    limit = PLATFORM_CHAR_LIMITS[PLATFORM]

    # Tighten toward the soft punchy-length target first (max 2 attempts, plain text only) —
    # this runs even when well under the 3000 hard cap, since the goal is short, not just legal.
    for shorten_attempt in range(2):
        if len(post) <= TARGET_POST_LENGTH:
            break
        print(f"  Post is {len(post)} chars — tightening toward ~{TARGET_POST_LENGTH} (attempt {shorten_attempt + 1}/2)...")
        shorten_prompt = (
            f"This LinkedIn post is {len(post)} characters. Tighten it to roughly {TARGET_POST_LENGTH} "
            f"characters (a bit under is fine) while keeping the hook, the strongest 2-3 points, and the "
            f"closing line. Cut everything else — repeated ideas, weaker bullets, filler sentences.\n"
            f"Plain text only — no markdown, no JSON wrapper.\n\n"
            f"Original post:\n{post}\n\n"
            f"Output ONLY the shortened post text. Nothing else."
        )
        post = generate_text(shorten_prompt, SYSTEM_PROMPT)
        post = _re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', post)
        post = _re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', post)
        post = post.strip()

    # If it's still over the real hard limit (rare after the tightening pass above), force it down
    for shorten_attempt in range(2):
        if len(post) <= limit:
            break
        print(f"  Post is {len(post)} chars — asking model to shorten (attempt {shorten_attempt + 1}/2)...")
        shorten_prompt = (
            f"This LinkedIn post is {len(post)} characters, over the {limit}-character limit.\n\n"
            f"Shorten it to strictly under {limit - 50} characters while keeping the hook, "
            f"story, insights, and CTA. Cut filler words, not ideas.\n"
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
    print(f"  Characters : {len(post)}/{limit}")
    print(f"  1st comment: {first_comment}\n")

    validate_post_length(post, PLATFORM)
    return post, first_comment


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2.5 — Humanize (rewrite so it reads like a real person, not an AI)
# ══════════════════════════════════════════════════════════════════════════════

HUMANIZE_SYSTEM_PROMPT = """
You are an expert human editor.

Your job is to rewrite the text below so it sounds like it was written by a real, thoughtful human — NOT by an AI.

IMPORTANT:
Do not merely replace words with synonyms. Rewrite the thinking, rhythm, sentence structure, and flow.

### REMOVE AI SLOP

Aggressively remove:

* Generic introductions
* "In today's fast-paced world..."
* "In the ever-evolving landscape..."
* "It's important to note that..."
* "Whether you're a beginner or an expert..."
* "Let's dive in..."
* "Here's the thing..."
* "The key takeaway is..."
* "At the end of the day..."
* "This isn't just X, it's Y", "X isn't Y, it's Z", "The real X isn't Y, it's Z", "Not X, but Y" — every
  form of negative parallelism. If the post needs one such contrast to make its point, keep at most ONE
  in the whole piece and rewrite it as two plain declarative sentences, not the "X, it's Y" template.
* Reveal bridges: "The result?", "Plot twist:", "Spoiler:", "Here's what/how/why happened" — delete the
  bridge and let the next sentence stand as a flat statement. It was the point anyway.
* Generic closing lines: "What do you think?", "Thoughts?", "Agree or disagree?", "Let me know in the
  comments", "Tag someone who needs this" — a closing question must name the specific thing this post
  is about, not float free of it.
* 2026 over-selected vocabulary (LLMs pick these 2-5x human rate — one is invisible, three in a
  paragraph is a signature): significant, crucial, notably, particularly, comprehensive, insights,
  robust, leverage, foster, landscape, nuanced, multifaceted, holistic, streamline, elevate, empower,
  utilize, facilitate, harness, unlock, navigate, seamless, ecosystem. Replace with the plain word
  (leverage→use, utilize→use, facilitate→help, streamline→simplify) or a real number where the vague
  word was standing in for one ("significant growth" → "31% growth").
* Present-participial clause openers ("Leveraging our data, we..." / "Building on this, ...") — put the
  actor first instead: "We used our data to...".
* Nominalisations that hide the actor ("the implementation of", "the utilization of") — use the verb:
  "when we implemented" instead of "the implementation of".
* Fake enthusiasm
* Corporate/LinkedIn language
* Unnecessary motivational language
* Repetitive conclusions
* Obvious summaries of what was just said
* Excessive headings
* Excessive bullet points
* Artificial transitions
* Overly polished sentences
* Needless adjectives and adverbs
* Repetitive sentence patterns
* "Furthermore", "Moreover", "Additionally", "However" when they aren't genuinely needed
* Generic claims such as "This can revolutionize..."
* Empty phrases that sound impressive but say nothing

Em dashes are not a tell by themselves — zero em dashes in a post that wanted one is itself a sign of
someone trying too hard to look human. Leave roughly one per 100 words; only trim the excess over that
(3+ in a short post is the real tell), and never replace one with a period — that creates fragment
stacking, which reads worse than the dash did.

### MAKE IT SOUND HUMAN

Use:

* Natural sentence lengths
* Contractions where appropriate
* Casual phrasing when the context allows it
* Specific examples instead of vague claims
* Opinions when the original writer clearly has one
* Natural transitions
* Direct language
* Concrete words
* A conversational tone
* Personality without forcing jokes
* Confidence without sounding like a marketing brochure

Don't make every sentence perfectly structured.

Real people don't write like textbooks.

Do NOT manufacture rhythm as a "sound human" tactic — deliberately mixing short punchy fragments with
long sentences, or forcing a staccato "Short. Punchy. Done." run, is itself a detectable, learnable
pattern once it's mechanical. Only fix a paragraph that reads genuinely machine-flat (four or more
sentences all nearly the same length, no subordinate clause anywhere). If the input already reads like
a person — one long sentence next to a short one here and there — leave its rhythm alone entirely.

### PRESERVE THE ORIGINAL THINKING

Do NOT:

* Change the meaning
* Invent facts
* Add information that wasn't there
* Remove important technical details
* Change numbers, names, examples, or claims
* Turn a simple explanation into something complicated
* Make the writing unnecessarily informal

Keep the author's actual ideas.

Improve how those ideas are expressed.

### IMPORTANT RULE

Don't try to "sound human" by deliberately adding mistakes.

No fake typos.

No unnecessary slang.

No forced humor.

No random "honestly", "literally", "basically", etc.

Human writing comes from natural thought and clear expression — not manufactured imperfections.

### STYLE TEST

Before returning the final version, ask yourself:

"If I saw this on the internet, would I immediately think an AI generated it?"

If the answer is yes, rewrite it again.

Then ask:

"Does this sound like one specific person actually had something to say?"

If not, rewrite it again.

### PLATFORM CONSTRAINTS (do not break these)

* Plain text only — no markdown formatting (no *, no _, no # headings).
* If you keep any list-style lines, prefix each with "→ " — never "*" or "-" — LinkedIn renders
  those as literal characters, not formatting.
* Do not add hashtags.
* Do not spell out any URL or domain name.
* The input is already written short and tight on purpose. Match its length or come in shorter —
  never pad it out with extra sentences, examples, or commentary that weren't in the original.

### FINAL OUTPUT

Return ONLY the rewritten content.

Do not explain what you changed.

Do not mention AI detection.

Do not mention this prompt.
""".strip()


def humanize_post(post_text: str) -> str:
    """Run the generated post through a human-editor rewrite pass so it reads like one
    specific person wrote it, not an AI. Preserves meaning/facts — only rewrites phrasing,
    rhythm, and sentence structure."""
    import re as _re

    print("[ Step 2.5 ] Humanizing post...")
    rewritten = generate_text(post_text, HUMANIZE_SYSTEM_PROMPT).strip()

    # Strip markdown fences/formatting the rewrite might reintroduce
    rewritten = _re.sub(r'^```(?:\w+)?\s*', '', rewritten)
    rewritten = _re.sub(r'\s*```$', '', rewritten)
    rewritten = _re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', rewritten)
    rewritten = _re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', rewritten)
    rewritten = _re.sub(r'^\s*\*\s+', '→ ', rewritten, flags=_re.MULTILINE)
    rewritten = rewritten.strip()

    limit = PLATFORM_CHAR_LIMITS[PLATFORM]

    # If the rewrite pushed it over the limit, ask the model to trim (same pattern as generate_post)
    for shorten_attempt in range(2):
        if len(rewritten) <= limit:
            break
        print(f"  Humanized post is {len(rewritten)} chars — asking model to shorten (attempt {shorten_attempt + 1}/2)...")
        shorten_prompt = (
            f"This LinkedIn post is {len(rewritten)} characters, over the {limit}-character limit.\n\n"
            f"Shorten it to strictly under {limit - 50} characters while keeping the same voice, "
            f"ideas, and specific details. Cut filler, not substance.\n"
            f"Plain text only — no markdown.\n\n"
            f"Original post:\n{rewritten}\n\n"
            f"Output ONLY the shortened post text. Nothing else."
        )
        rewritten = generate_text(shorten_prompt, HUMANIZE_SYSTEM_PROMPT)
        rewritten = _re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', rewritten)
        rewritten = _re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', rewritten)
        rewritten = rewritten.strip()

    if len(rewritten) > limit:
        print("  Humanized shortening did not converge — applying hard truncation.")
        rewritten = truncate_for_platform(rewritten, PLATFORM)

    print(f"\n  Humanized post:\n  {'─'*50}")
    for line in rewritten.split("\n"):
        print(f"  {line}")
    print(f"  {'─'*50}")
    print(f"  Characters : {len(rewritten)}/{limit}\n")

    validate_post_length(rewritten, PLATFORM)
    return rewritten


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2.6 — Generate & render infographic (optional, graceful fallback)
# ══════════════════════════════════════════════════════════════════════════════

INCLUDE_INFOGRAPHIC = os.environ.get("INCLUDE_INFOGRAPHIC", "1") == "1"
# Dark outperforms light in-feed (stands out against LinkedIn's white UI) per current engagement
# data — set INFOGRAPHIC_THEME=light to switch back without a code change.
INFOGRAPHIC_THEME = os.environ.get("INFOGRAPHIC_THEME", "dark").strip().lower()
_INFOGRAPHIC_TEMPLATE = (
    "process_infographic_dark.html.j2" if INFOGRAPHIC_THEME == "dark" else "process_infographic.html.j2"
)
_PNG_PATH = os.path.join(_script_dir, "..", "renderer", "output", "infographic.png")


def build_infographic(topic: str, post_text: str) -> str | None:
    """Generate infographic content (grounded in the actual post text) + render PNG using the
    'how it works' process template, in the configured theme (dark by default). Returns local
    PNG path, or None on failure."""
    if not INCLUDE_INFOGRAPHIC:
        return None

    try:
        import sys as _sys, pathlib as _pl
        _root = str(_pl.Path(__file__).parent.parent)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        import scripts.infographic as ig
    except ImportError:
        print("  [infographic] skipped — scripts.infographic not importable.")
        return None

    print(f"[ Step 2.6 ] Generating infographic (synced to post, template: process/how-it-works, theme: {INFOGRAPHIC_THEME})...")
    try:
        content = ig.generate_process_content(topic, post_text, generate_text)
        png     = ig.render_infographic(content, _PNG_PATH, template=_INFOGRAPHIC_TEMPLATE)
        print(f"  Infographic rendered: {png}\n")
        return png
    except Exception as exc:
        print(f"  [infographic] WARNING: failed ({exc}) — posting text-only.\n")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Post directly to LinkedIn
# ══════════════════════════════════════════════════════════════════════════════

def post_to_linkedin(post_text: str, image_urn: str = None) -> str:
    """Publish the post directly to LinkedIn. Attaches image if image_urn provided."""
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

    if image_urn:
        share_content = {
            "shareCommentary":    {"text": post_text},
            "shareMediaCategory": "IMAGE",
            "media": [{"status": "READY", "media": image_urn}],
        }
    else:
        share_content = {
            "shareCommentary":    {"text": post_text},
            "shareMediaCategory": "NONE",
        }

    payload = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": share_content,
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
# STEP 4 — Drop a punchy first comment to seed engagement
# ══════════════════════════════════════════════════════════════════════════════
# Kept as a separate comment (rather than baked into the post) so it reads as
# the author's own reply — a common tactic to kickstart replies on the post.

def post_engagement_comment(post_id: str, comment_text: str) -> None:
    """Add the author's own first comment to the just-published post. Non-fatal on failure."""
    if not comment_text:
        return

    print("[ Step 4 ] Dropping first comment to seed engagement...")

    share_urn = post_id if post_id.startswith("urn:") else f"urn:li:ugcPost:{post_id}"
    encoded_urn = requests.utils.quote(share_urn, safe="")

    response = requests.post(
        f"https://api.linkedin.com/v2/socialActions/{encoded_urn}/comments",
        headers={
            "Authorization":             f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "Content-Type":              "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        },
        json={
            "actor": f"urn:li:person:{LINKEDIN_PERSON_ID}",
            "message": {"text": comment_text},
        },
        timeout=15,
    )

    if response.status_code in (200, 201):
        print(f"  Comment posted.\n")
    else:
        try:
            err = response.json()
        except ValueError:
            err = response.text
        print(f"  [warn] Could not post first comment ({response.status_code}): {err}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(preview: bool = False):
    topic_override = os.environ.get("TOPIC_OVERRIDE", "").strip()
    topic = topic_override if topic_override else random.choice(TOPICS)
    tone  = random.choice(TONES)

    print(f"\n{'='*60}")
    print(f"  LinkedIn Post Agent — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    if preview:
        print(f"  MODE: PREVIEW (no LinkedIn posting)")
    print(f"{'='*60}")
    print(f"  Slot  : [{CONTENT_SLOT.upper()}] {SLOT_LABEL}")
    print(f"  Topic : {topic}" + (" (forced via TOPIC_OVERRIDE)" if topic_override else ""))
    print(f"  Tone  : {tone}")
    print(f"{'='*60}\n")

    try:
        research = research_topic(topic, NICHE)
        draft, first_comment = generate_post(topic, tone, NICHE, PERSONA, research)
        post     = humanize_post(draft)
        png_path = build_infographic(topic, post)

        if preview:
            print(f"  Would comment: {first_comment}\n")
            print(f"{'='*60}")
            print(f"  PREVIEW ONLY — post NOT published to LinkedIn.")
            if png_path:
                print(f"  Infographic preview saved to: {png_path}")
            print(f"  Run without --preview to publish it.")
            print(f"{'='*60}\n")
            return

        validate_post_length(post, PLATFORM)

        image_urn = None
        if png_path:
            try:
                import scripts.infographic as ig
                image_urn = ig.upload_to_linkedin(png_path, LINKEDIN_ACCESS_TOKEN, LINKEDIN_PERSON_ID)
            except Exception as exc:
                print(f"  [infographic] WARNING: upload failed ({exc}) — posting text-only.\n")

        post_id = post_to_linkedin(post, image_urn=image_urn)
        post_engagement_comment(post_id, first_comment)

        print(f"{'='*60}")
        print(f"  Done! Post published directly to LinkedIn.")
        print(f"  Image attached    : {'yes' if image_urn else 'no (text-only)'}")
        print(f"  LinkedIn Post ID : {post_id}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n  ERROR: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    main(preview="--preview" in sys.argv)
