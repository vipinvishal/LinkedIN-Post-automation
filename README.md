# LinkedIn Post Automation

An AI agent that researches trending topics, generates engaging posts, and publishes them directly to LinkedIn — 4× every day.

**No VPS needed. No manual work. Fully automated via GitHub Actions.**

---

## How It Works

```
GitHub Actions (9 AM / 1 PM / 6 PM / 10 PM IST)
        ↓
Exa — neural web research on a random AI/tech topic
        ↓
Gemini — generates a viral, first-person post
  └─ fallback: Gemini key #2 → Euron API
        ↓
LinkedIn API — publishes directly to your profile
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **GitHub Actions** | 4× daily scheduling (replaces VPS/cron) |
| **Exa** | Real-time neural web research |
| **Google Gemini** | Post generation (dual-key with quota rotation) |
| **Euron API** | Fallback when all Gemini keys are exhausted |
| **LinkedIn UGC API** | Direct publishing to LinkedIn |

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
```

### 3. Set up your `.env` file

```bash
cp .env.example .env
```

Fill in your API keys (see [Configuration](#configuration) below).

### 4. Test locally before going live

```bash
# Preview a generated post without posting to LinkedIn
python scripts/generate_and_schedule.py --preview

# Run the full pipeline (research → generate → post to LinkedIn)
python scripts/generate_and_schedule.py
```

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
**GitHub repo → Actions → Daily LinkedIn Post → Run workflow**

---

## Customizing Topics & Persona

Edit `scripts/topics.json` to change:

- **`niche`** — the content category
- **`persona`** — the voice and style of the posts
- **`content_slots`** — topics and tones for each time slot
- **`portfolio_url`** — if set, this URL is auto-posted as the first comment on every
  published post (kept out of the post body so it doesn't suppress LinkedIn's organic reach)

---

## Project Structure

```
├── scripts/
│   ├── generate_and_schedule.py   # main pipeline
│   ├── topics.json                # niche, topics, tones, persona
│   └── get_linkedin_token.py      # one-time helper to get LinkedIn tokens
├── .github/
│   └── workflows/
│       └── daily_post.yml         # GitHub Actions workflow
├── .env.example                   # template — copy to .env and fill in keys
├── requirements.txt               # Python dependencies
└── .gitignore
```

---

## Fallback Chain

If Gemini hits its daily free-tier quota, the bot automatically falls back:

```
Gemini key #1 → Gemini key #2 → Euron API (gemini-2.0-flash)
```

No manual intervention needed.

---

## License

MIT
