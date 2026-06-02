import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['soprahr_rag']
    count = await db['tickets'].count_documents({})
    print('Total tickets:', count)
    doc = await db['tickets'].find_one({})
    if doc:
        print('Sample reference:', doc.get('reference'))
    results = await db['tickets'].find({'reference': {'$regex': 'FR W210001', '$options': 'i'}}).to_list(None)
    print('Regex matches for FR W210001:', len(results))
    if results:
        print('Matched reference:', results[0].get('reference'))
    client.close()

asyncio.run(check())
