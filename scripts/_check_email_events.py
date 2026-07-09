"""Temporary debug helper: checks Brevo delivery events + blocked list for SopraHR admin email."""
import os, json, ssl, http.client, certifi
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("BREVO_API_KEY", "")
target = "my.zaidi@soprahr.com"
encoded = "my.zaidi%40soprahr.com"

ctx = ssl.create_default_context(cafile=certifi.where())
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def brevo_get(path):
    conn = http.client.HTTPSConnection("api.brevo.com", context=ctx, timeout=30)
    conn.request("GET", path, headers={"accept": "application/json", "api-key": key})
    r = conn.getresponse()
    data = json.loads(r.read())
    conn.close()
    return r.status, data


print("=== EMAIL EVENTS ===")
status, data = brevo_get(f"/v3/smtp/statistics/events?email={encoded}&limit=20")
for e in data.get("events", []):
    date = e.get("date", "")[:19]
    event = e.get("event", "")
    msg_id = (e.get("messageId") or "")[-20:]
    reason = e.get("reason") or ""
    print(f"  {date}  {event:<12}  {msg_id}  {reason}")

print()
print("=== BLOCKED CONTACTS ===")
status2, data2 = brevo_get(f"/v3/contacts/blockedContacts?email={encoded}&limit=5")
blocked = data2.get("contacts", [])
if not blocked:
    print("  Not in blocked list")
else:
    for c in blocked:
        print(" ", c)

print()
print("=== SPAM COMPLAINTS ===")
status3, data3 = brevo_get(f"/v3/contacts/complaints?startDate=2026-06-01&endDate=2026-06-09&limit=10")
complaints = [c for c in data3.get("complaints", []) if target in str(c)]
print(f"  {len(complaints)} complaint(s) for {target}")
for c in complaints:
    print(" ", c)
