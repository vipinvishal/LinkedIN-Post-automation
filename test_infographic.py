#!/usr/bin/env python3
"""
Local infographic preview — generates with Gemini, renders PNG, opens it.
Does NOT post to LinkedIn.

Usage:
  python test_infographic.py
  python test_infographic.py "RAG systems in production"
"""
import os, sys, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

TOPIC = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How to build AI agents that actually work in production"

SAMPLE_RESEARCH = (
    "AI agents fail in production due to poor memory management, brittle tool calling, "
    "lack of error handling, and no observability. Key fixes: structured outputs, "
    "guardrails, retry logic, logging every LLM call, and testing edge cases early."
)

OUT = str(Path(__file__).parent / "renderer" / "output" / "preview.png")


def main():
    from scripts.generate_and_schedule import generate_text
    import scripts.infographic as ig

    print(f"\nTopic : {TOPIC}")
    print("Generating infographic content via Gemini...\n")

    content = ig.generate_content(TOPIC, SAMPLE_RESEARCH, generate_text)

    print(f"  title : {content['title_line1']} / {content['title_line2']}")
    for k in ["box1", "box2", "box3", "box4", "box5"]:
        print(f"  {k}   : [{content[k]['label']}] {content[k]['points']}")

    print("\nRendering PNG...")
    ig.render_infographic(content, OUT)
    print(f"Saved → {OUT}\n")
    subprocess.run(["open", OUT])


if __name__ == "__main__":
    main()
