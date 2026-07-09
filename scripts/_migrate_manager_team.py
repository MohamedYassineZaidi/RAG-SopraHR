import asyncio, os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
load_dotenv()

async def migrate():
    c = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = c[os.getenv("MONGODB_DB", "soprahr_rag")]
    r = await db["users"].update_many({"role": "MANAGER"}, {"$set": {"team": None}})
    print(f"Matched: {r.matched_count}, Modified: {r.modified_count}")
    managers = await db["users"].find({"role": "MANAGER"}, {"email": 1, "team": 1}).to_list(None)
    for m in managers:
        print(f"  {m['email']} -> team={m.get('team')}")
    c.close()

asyncio.run(migrate())
