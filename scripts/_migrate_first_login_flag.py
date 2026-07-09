import asyncio
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

async def main() -> None:
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGODB_DB", "soprahr_rag")]

    never_logged = await db["users"].update_many(
        {"must_change_password": {"$exists": False}, "last_login": None},
        {"$set": {"must_change_password": True}},
    )
    already_logged = await db["users"].update_many(
        {"must_change_password": {"$exists": False}, "last_login": {"$ne": None}},
        {"$set": {"must_change_password": False}},
    )

    print(
        f"first-login flagged: matched={never_logged.matched_count}, modified={never_logged.modified_count}"
    )
    print(
        f"already-logged flagged: matched={already_logged.matched_count}, modified={already_logged.modified_count}"
    )

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
