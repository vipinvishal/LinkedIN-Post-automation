#!/usr/bin/env python3
"""
One-time LinkedIn OAuth token generator.

Run this once to get your access token + person ID, then store them
as GitHub secrets: LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_ID.

Usage:
    python scripts/get_linkedin_token.py

Requirements:
    LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET must be set in .env
"""

import os
import sys
import secrets
import webbrowser
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

CLIENT_ID     = os.environ.get("LINKEDIN_CLIENT_ID")
CLIENT_SECRET = os.environ.get("LINKEDIN_CLIENT_SECRET")
REDIRECT_URI  = "http://localhost:8080/callback"
SCOPES        = "openid profile w_member_social email"

if not CLIENT_ID or not CLIENT_SECRET:
    sys.exit(
        "ERROR: LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET must be set in your .env file.\n"
        "  Find them at: https://developer.linkedin.com → your app → Auth tab"
    )

# ── Step 1: build the auth URL ────────────────────────────────────────────────

state = secrets.token_urlsafe(16)

auth_url = (
    "https://www.linkedin.com/oauth/v2/authorization?"
    + urllib.parse.urlencode({
        "response_type": "code",
        "client_id":     CLIENT_ID,
        "redirect_uri":  REDIRECT_URI,
        "scope":         SCOPES,
        "state":         state,
    })
)

# ── Step 2: local callback server ─────────────────────────────────────────────

_auth_code = None
_got_state = None

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _auth_code, _got_state
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        _auth_code = params.get("code", [None])[0]
        _got_state = params.get("state", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>Done! You can close this tab and return to the terminal.</h2>")

    def log_message(self, *args):
        pass  # silence request logs

# ── Step 3: open browser + wait for callback ──────────────────────────────────

print("\n" + "="*60)
print("  LinkedIn OAuth — one-time token setup")
print("="*60)
print("\n  Opening your browser for LinkedIn authorization...")
print("  (If it does not open automatically, paste this URL:)")
print(f"\n  {auth_url}\n")

webbrowser.open(auth_url)

server = HTTPServer(("localhost", 8080), CallbackHandler)
server.handle_request()  # blocks until one request arrives

if not _auth_code:
    sys.exit("ERROR: No authorization code received. Did you approve the app?")

if _got_state != state:
    sys.exit("ERROR: State mismatch — possible CSRF. Run the script again.")

print("  Authorization code received.\n")

# ── Step 4: exchange code for access token ────────────────────────────────────

resp = requests.post(
    "https://www.linkedin.com/oauth/v2/accessToken",
    data={
        "grant_type":    "authorization_code",
        "code":          _auth_code,
        "redirect_uri":  REDIRECT_URI,
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    },
    timeout=15,
)
resp.raise_for_status()
token_resp = resp.json()

access_token = token_resp.get("access_token")
expires_in   = token_resp.get("expires_in", 0)
expires_days = round(expires_in / 86400)

if not access_token:
    sys.exit(f"ERROR: Token exchange failed: {token_resp}")

print(f"  Access token received (expires in ~{expires_days} days).\n")

# ── Step 5: fetch person ID from userinfo ─────────────────────────────────────

resp = requests.get(
    "https://api.linkedin.com/v2/userinfo",
    headers={"Authorization": f"Bearer {access_token}"},
    timeout=15,
)
resp.raise_for_status()
userinfo = resp.json()

person_id   = userinfo.get("sub")
person_name = userinfo.get("name", "unknown")

if not person_id:
    sys.exit(f"ERROR: Could not retrieve person ID: {userinfo}")

# ── Step 6: print the secrets ─────────────────────────────────────────────────

print("="*60)
print(f"  Authenticated as: {person_name}")
print("="*60)
print()
print("  Copy these two values into your GitHub secrets")
print("  (Settings → Secrets and variables → Actions → New secret):")
print()
print(f"  Secret name : LINKEDIN_ACCESS_TOKEN")
print(f"  Secret value: {access_token}")
print()
print(f"  Secret name : LINKEDIN_PERSON_ID")
print(f"  Secret value: {person_id}")
print()
print("  Also add them to your local .env:")
print(f"  LINKEDIN_ACCESS_TOKEN={access_token}")
print(f"  LINKEDIN_PERSON_ID={person_id}")
print()
print(f"  Token expires in ~{expires_days} days. Re-run this script to refresh.")
print("="*60 + "\n")
