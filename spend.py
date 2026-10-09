"""Prints what this OpenRouter key has spent so far. Free to call; never prints the key."""
import json, os, urllib.request
here = os.path.dirname(os.path.abspath(__file__))
key = [l.split("=", 1)[1].strip() for l in open(os.path.join(here, ".env")).read().splitlines() if l.startswith("OPENROUTER_API_KEY=")][0]
d = json.load(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key})))["data"]
print("limit $%s | used $%.4f | remaining $%s | free tier: %s" % (d.get("limit"), d.get("usage") or 0, d.get("limit_remaining"), d.get("is_free_tier")))
