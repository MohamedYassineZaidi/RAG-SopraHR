import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

async def migrate():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
    result = await db["users"].update_many(
        {"role": "ADMIN"},
        {"$set": {"team": None}}
    )
    print(f"Matched: {result.matched_count}, Modified: {result.modified_count}")
    admins = await db["users"].find({"role": "ADMIN"}, {"email": 1, "team": 1}).to_list(None)
    for a in admins:
        print(f"  {a['email']} -> team={a.get('team')}")
    client.close()

asyncio.run(migrate())
