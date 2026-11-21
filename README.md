# LinkedIn Post Automation

An AI agent that researches trending AI/ML topics, writes a short, hook-driven post in one engineer's
learning-in-public voice, generates a matching infographic, and publishes both directly to LinkedIn —
4× every day, fully unattended.

**No VPS needed. No manual work. Fully automated via GitHub Actions.**

---

## How It Works

```
GitHub Actions (9 AM / 1 PM / 6 PM / 10 PM IST)
        ↓
Exa — neural web research on a random AI/tech topic
        ↓
Gemini — generates a post using a hook formula from the hook matrix
  └─ fallback: Gemini key #2 → Euron API
        ↓
Gemini — humanizes the draft (strips AI-sounding tells, tightens length)
        ↓
Gemini + Playwright — generates a matching infographic (dark or light theme)
        ↓
LinkedIn API — publishes the post + infographic directly to your profile
        ↓
LinkedIn API — drops a topic-specific engagement comment (the author's own "first reply")
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **GitHub Actions** | 4× daily scheduling (replaces VPS/cron) |
| **Exa** | Real-time neural web research |
| **Google Gemini** | Post generation, humanizing, and infographic content — dual-key with quota rotation, auto-falls-through if a model gets retired |
| **Euron API** | Fallback when all Gemini keys are exhausted |
| **Playwright + Jinja2** | Renders the infographic (HTML/CSS template → PNG) |
| **LinkedIn UGC API** | Direct publishing of the post, image, and first comment |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/vipinvishal/LinkedIN-Post-automation.git
cd LinkedIN-Post-automation
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium   # needed to render the infographic locally
```

### 3. Set up your `.env` file

```bash
cp .env.example .env
```

Fill in your API keys (see [Configuration](#configuration) below).

### 4. Test locally before going live

```bash
# Preview a generated post + infographic without posting to LinkedIn
python scripts/generate_and_schedule.py --preview

# Run the full pipeline (research → generate → humanize → infographic → post → comment)
python scripts/generate_and_schedule.py
```

The generated infographic is written to `renderer/output/infographic.png` on every run.

---

## Configuration

Add these to your `.env` file:

| Variable | Where to get it | Required |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Yes |
| `GEMINI_API_KEY_2` | Same — second Google account | Optional (quota fallback) |
| `EURON_API_KEY` | [euron.one](https://euron.one) | Optional (last-resort fallback) |
| `EXA_API_KEY` | [exa.ai](https://exa.ai) | Yes |
| `LINKEDIN_ACCESS_TOKEN` | Run `python scripts/get_linkedin_token.py` | Yes |
| `LINKEDIN_PERSON_ID` | Run `python scripts/get_linkedin_token.py` | Yes |
| `GEMINI_MODEL` | Overrides the default model (`gemini-2.5-flash`) | Optional |
| `INCLUDE_INFOGRAPHIC` | Set to `0` to skip infographic generation entirely | Optional (default `1`) |
| `INFOGRAPHIC_THEME` | `dark` (default) or `light` | Optional |

LinkedIn's OAuth token expires roughly every 60 days — when posting starts failing with a 401, re-run
`python scripts/get_linkedin_token.py` and update the secret/env var.

---

## GitHub Actions Setup (Automated Daily Posting)

### 1. Add secrets to your GitHub repo

Go to **Settings → Secrets and variables → Actions → New repository secret** and add:

- `GEMINI_API_KEY`
- `GEMINI_API_KEY_2`
- `EURON_API_KEY`
- `EXA_API_KEY`
- `LINKEDIN_ACCESS_TOKEN`
- `LINKEDIN_PERSON_ID`

### 2. The workflow runs automatically

The workflow is defined in `.github/workflows/daily_post.yml` and triggers 4× daily:

| Time (IST) | Content Slot |
|---|---|
| 9:00 AM | Breaking AI news / hot take |
| 1:00 PM | AI educational post |
| 6:00 PM | Personal learning / build-in-public |
| 10:00 PM | Advanced AI concept |

You can also trigger it manually anytime:
**GitHub repo → Actions → Daily LinkedIn Post → Run workflow** (check **preview** to test without publishing)

---

## Content Pipeline

Every run follows the same six steps (see `scripts/generate_and_schedule.py::main()`):

1. **Research** — Exa pulls 5 recent sources on a topic drawn from the active content slot.
2. **Generate** — Gemini writes the post using one of two body structures (problem→solution or
   scenario→risk→solution, alternated deterministically by day), with the opening line driven by
   a formula picked from the [hook matrix](#hook-matrix). Target length: 900-1,300 characters —
   short enough to read in one glance while scrolling, not an essay.
3. **Humanize** — a second pass strips AI-sounding vocabulary, reveal-bridge phrasing, negative
   parallelism, and manufactured sentence-rhythm tricks, without changing any fact, number, or claim.
4. **Infographic** — Gemini turns the *same* post into structured "how it works" content (stages,
   4 numbered steps, a closing hook), rendered to PNG via a Jinja2/Playwright template. Because it's
   generated from the final post text, it's always in sync with whatever hook/topic that post used.
5. **Post** — publishes the text + infographic directly via the LinkedIn UGC API.
6. **Engagement comment** — drops a short, topic-specific comment (also model-generated, in the
   author's voice) as the post's first reply, to seed discussion.

### Hook Matrix

`scripts/hook_matrix.py` is a library of scroll-stopping opener formulas built specifically for this
persona (an individual engineer learning AI/ML in public — not a founder or business voice). It covers:

- **Pattern interrupts** — e.g. "The Broken Assumption", "The Silent Change"
- **Psychological triggers** — competence-gap, insider-knowledge, relatable-struggle, specificity-as-authority
- **Curiosity gaps** — must resolve within ~210 characters (LinkedIn's "see more" cutoff), with a
  banned-phrase list for manufactured-curiosity tells
- **Power phrases** — grounded, first-person connective lines, offered as optional seasoning
- **8 full hook structures** — Number-First Reveal, Time-Anchor Confession, Controlled A/B Anecdote,
  Curiosity-Gap Teaser, Contrarian + Dated Receipts, Anecdote-Meets-Evidence Bridge, Explain-While-
  Learning, False-Binary Dissolve

`select_hook_formula()` rotates through the matrix deterministically (day-of-year + slot index), so
every formula gets exercised evenly over time instead of clustering by chance.

### Infographics

Two themes live in `renderer/templates/`: `process_infographic_dark.html.j2` (default) and
`process_infographic.html.j2` (light). Switch with `INFOGRAPHIC_THEME=light` — no code change needed.
Both share the same layout (3-stage flow, 4 numbered cards, two "flow chip" summaries, a sticky-note
pull-quote) and the same violet/amber/green/magenta accent system, just recolored for contrast.

---

## Customizing Topics & Persona

Edit `scripts/topics.json` to change:

- **`niche`** — the content category
- **`persona`** — the voice and style of the posts
- **`content_slots`** — topics and tones for each time slot

Edit `scripts/hook_matrix.py` to add, remove, or retune hook formulas — each entry has a `best_for`
list of content slots it's tagged for.

---

## Project Structure

```
├── scripts/
│   ├── generate_and_schedule.py   # main pipeline (research → generate → humanize → post → comment)
│   ├── infographic.py             # infographic content generation + LinkedIn image upload
│   ├── hook_matrix.py             # niche-specific hook formula library
│   ├── topics.json                # niche, topics, tones, persona
│   └── get_linkedin_token.py      # one-time / periodic helper to (re-)get LinkedIn tokens
├── renderer/
│   ├── render.py                  # Jinja2 → Playwright PNG renderer
│   └── templates/
│       ├── process_infographic.html.j2        # light theme
│       └── process_infographic_dark.html.j2   # dark theme (default)
├── .github/
│   └── workflows/
│       └── daily_post.yml         # GitHub Actions workflow
├── test_infographic.py            # exercises the live infographic-generation path
├── .env.example                   # template — copy to .env and fill in keys
├── requirements.txt               # Python dependencies
└── .gitignore
```

---

## Fallback Chain

If a model gets retired or a key hits its daily quota, the bot automatically falls through:

```
gemini-flash-latest → gemini-3.6-flash → gemini-2.5-flash  (key #1)
        → same model list on key #2
        → Euron API (gemini-3.6-flash)
```

A 404 "model not found" (e.g. a future Gemini retirement) is treated the same as a quota error — it
tries the next model automatically instead of failing the whole run. No manual intervention needed
unless every fallback is exhausted, in which case the run fails loudly with a clear error.

---

## License

MIT
