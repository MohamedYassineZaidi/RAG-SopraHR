#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
database.py
===========
MongoDB async operations for users, tickets, and analysis history.

Usage:
  from scripts.database import init_db, get_ticket_by_reference, search_tickets
  
  await init_db()
  ticket = await get_ticket_by_reference("FR W210000")
"""

import os
from datetime import datetime
from typing import Optional, List
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING, TEXT, ASCENDING, UpdateOne
from bson import ObjectId, BSON

# MongoDB hard limit per BSON document. Leave a small safety margin so that
# server-side fields (e.g. _id additions) do not push us over the edge.
_MAX_BSON_BYTES = 16 * 1024 * 1024
_BSON_SAFETY_MARGIN = 64 * 1024  # 64 KB

from scripts.models import (
    Ticket, TicketCreate, TicketSearch,
    User, UserCreate, UserProfile, AdminUserCreate,
    AnalysisHistory, AnalysisHistoryCreate,
    PipelineJobCreate, PipelineJob, PipelineJobLog,
    EvalRunCreate, AuditLogCreate, NotificationCreate,
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _str_id(doc: dict) -> dict:
    """Convert MongoDB ObjectId _id to string in-place."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc
# ─────────────────────────────────────────────

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[dict] = None


# ─────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────

async def init_db() -> None:
    """Initialize MongoDB connection and create indexes."""
    global _client, _db
    
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGODB_DB", "soprahr_rag")
    
    _client = AsyncIOMotorClient(mongo_url)
    _db = _client[mongo_db]
    
    # Test connection
    try:
        await _client.admin.command("ping")
        print(f"[OK] Connected to MongoDB: {mongo_db}")
    except Exception as e:
        print(f"[ERROR] MongoDB connection failed: {e}")
        raise
    
    # Create indexes
    await _create_indexes()


async def close_db() -> None:
    """Close MongoDB connection."""
    if _client:
        _client.close()
        print("[OK] MongoDB connection closed")


async def _create_indexes() -> None:
    """Create indexes for optimized queries."""
    # Tickets indexes
    tickets = _db["tickets"]
    await tickets.create_index("reference", unique=True)
    await tickets.create_index("support_team")
    await tickets.create_index("is_closed")
    await tickets.create_index([("description", TEXT), ("title", TEXT)])
    await tickets.create_index([("created_at", DESCENDING)])
    print("[OK] Ticket indexes created")
    
    # Users indexes
    users = _db["users"]
    await users.create_index("username", unique=True)
    await users.create_index("email", unique=True)
    await users.create_index("role")
    await users.create_index("team")
    print("[OK] User indexes created")
    
    # Analysis history indexes
    history = _db["analysis_history"]
    await history.create_index("user_id")
    await history.create_index([("created_at", DESCENDING)])
    await history.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    print("[OK] Analysis history indexes created")
    # Pipeline jobs indexes
    pipeline_jobs = _db["pipeline_jobs"]
    await pipeline_jobs.create_index("job_id", unique=True)
    await pipeline_jobs.create_index([("started_at", DESCENDING)])
    await pipeline_jobs.create_index("status")
    await pipeline_jobs.create_index("triggered_by")
    print("[OK] Pipeline jobs indexes created")
    # Evaluation runs indexes
    eval_runs = _db["eval_runs"]
    await eval_runs.create_index("run_id", unique=True)
    await eval_runs.create_index([("started_at", DESCENDING)])
    await eval_runs.create_index("status")
    await eval_runs.create_index("triggered_by")
    await eval_runs.create_index("eval_type")
    print("[OK] Evaluation runs indexes created")
    # Notifications indexes
    notifications = _db["notifications"]
    await notifications.create_index("user_id")
    await notifications.create_index("read")
    await notifications.create_index([("created_at", DESCENDING)])
    await notifications.create_index([("user_id", ASCENDING), ("read", ASCENDING), ("created_at", DESCENDING)])
    print("[OK] Notifications indexes created")

# ─────────────────────────────────────────────
# TICKET OPERATIONS
# ─────────────────────────────────────────────

async def create_ticket(ticket_data: TicketCreate) -> Ticket:
    """Create a new ticket."""
    tickets = _db["tickets"]
    now = datetime.utcnow()
    
    doc = {
        **ticket_data.dict(),
        "created_at": now,
        "updated_at": now,
    }
    
    result = await tickets.insert_one(doc)
    doc["_id"] = result.inserted_id
    
    return Ticket(**_str_id(doc))


async def get_ticket_by_reference(reference: str) -> Optional[Ticket]:
    """Fetch ticket by reference (e.g., 'FR W210000')."""
    tickets = _db["tickets"]
    doc = await tickets.find_one({"reference": reference})
    
    if doc:
        return Ticket(**_str_id(doc))
    return None


async def get_ticket_by_id(ticket_id: str) -> Optional[Ticket]:
    """Fetch ticket by MongoDB _id."""
    tickets = _db["tickets"]
    try:
        doc = await tickets.find_one({"_id": ObjectId(ticket_id)})
        if doc:
            return Ticket(**_str_id(doc))
    except:
        pass
    return None


async def search_tickets(
    query: str = "",
    support_team: Optional[str] = None,
    is_closed: Optional[bool] = None,
    limit: int = 20,
    skip: int = 0
) -> tuple[List[TicketSearch], int]:
    """
    Search tickets by keyword, team, or status.
    Returns: (results, total_count)
    """
    tickets = _db["tickets"]
    filter_dict = {}
    
    if support_team:
        filter_dict["support_team"] = support_team
    
    if is_closed is not None:
        filter_dict["is_closed"] = is_closed
    
    # Reference search (case-insensitive regex, e.g. "fr w2100" matches "FR W210001")
    if query:
        filter_dict["reference"] = {"$regex": query.strip(), "$options": "i"}
    
    # Get total count
    total_count = await tickets.count_documents(filter_dict)
    
    # Get paginated results
    results = await tickets.find(filter_dict).skip(skip).limit(limit).to_list(None)
    
    search_results = [
        TicketSearch(
            id=str(doc["_id"]),
            reference=doc.get("reference", ""),
            title=doc.get("title", ""),
            support_team=doc.get("support_team"),
            is_closed=doc.get("is_closed", False),
            description=doc.get("description", "")[:300],
        )
        for doc in results
    ]
    
    return search_results, total_count


async def get_recent_tickets(limit: int = 20) -> List[Ticket]:
    """Get recently updated tickets."""
    tickets = _db["tickets"]
    docs = await tickets.find().sort("updated_at", DESCENDING).limit(limit).to_list(None)
    return [Ticket(**_str_id(doc)) for doc in docs]


async def update_ticket(ticket_id: str, update_data: dict) -> Optional[Ticket]:
    """Update ticket fields."""
    tickets = _db["tickets"]
    update_data["updated_at"] = datetime.utcnow()
    
    try:
        result = await tickets.find_one_and_update(
            {"_id": ObjectId(ticket_id)},
            {"$set": update_data},
            return_document=True
        )
        if result:
            return Ticket(**_str_id(result))
    except:
        pass
    return None


async def bulk_insert_tickets(tickets_list: List[TicketCreate]) -> int:
    """Bulk upsert multiple tickets by reference.

    Skips individual documents that exceed MongoDB's 16 MB BSON limit so
    one oversized ticket cannot abort the whole import batch.
    """
    tickets = _db["tickets"]
    now = datetime.utcnow()

    operations = []
    oversized = 0
    size_limit = _MAX_BSON_BYTES - _BSON_SAFETY_MARGIN

    for ticket in tickets_list:
        # Convert to dict with proper nested object serialization
        ticket_dict = ticket.dict()

        # Ensure conversation entries are properly serialized
        if ticket_dict.get("conversation"):
            ticket_dict["conversation"] = [
                conv.dict() if hasattr(conv, "dict") else conv
                for conv in ticket_dict["conversation"]
            ]

        doc = {
            **ticket_dict,
            "updated_at": now,
            "created_at": now,
        }

        # Pre-flight BSON size check to avoid bulk_write aborts on giant
        # tickets (server error 17420 "Document to upsert is larger than
        # 16777216").
        try:
            doc_size = len(BSON.encode(doc))
        except Exception as e:
            print(f"[WARN] Skipping ticket {ticket.reference}: cannot encode to BSON ({e})")
            oversized += 1
            continue

        if doc_size > size_limit:
            print(
                f"[WARN] Skipping ticket {ticket.reference}: document size "
                f"{doc_size} bytes exceeds MongoDB limit ({size_limit} bytes); "
                f"messages={len(ticket_dict.get('conversation') or [])}"
            )
            oversized += 1
            continue

        # Strip created_at from $set so we don't overwrite it on updates;
        # it is restored via $setOnInsert below.
        doc.pop("created_at", None)

        operations.append(
            UpdateOne(
                {"reference": ticket.reference},
                {
                    "$set": doc,
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
        )

    if oversized:
        print(f"[INFO] Skipped {oversized} oversized ticket(s) above the 16MB BSON limit.")

    if not operations:
        return 0

    result = await tickets.bulk_write(operations, ordered=False)
    return result.upserted_count + result.modified_count


# ─────────────────────────────────────────────
# USER OPERATIONS
# ─────────────────────────────────────────────

async def create_user(
    user_data: UserCreate,
    password_hash: str,
    role: str = "CONSULTANT"
) -> User:
    """Create a new user."""
    users = _db["users"]
    now = datetime.utcnow()
    
    doc = {
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": password_hash,
        "role": role,
        "team": user_data.team,
        "created_at": now,
        "updated_at": now,
        "favorite_tickets": [],
        "is_active": True,
    }
    
    result = await users.insert_one(doc)
    doc["_id"] = result.inserted_id
    
    return User(**_str_id(doc))


async def get_user_by_username(username: str) -> Optional[User]:
    """Fetch user by username."""
    users = _db["users"]
    doc = await users.find_one({"username": username})
    
    if doc:
        return User(**_str_id(doc))
    return None


async def get_user_by_id(user_id: str) -> Optional[User]:
    """Fetch user by MongoDB _id."""
    users = _db["users"]
    try:
        doc = await users.find_one({"_id": ObjectId(user_id)})
        if doc:
            return User(**_str_id(doc))
    except:
        pass
    return None


async def get_user_by_email(email: str) -> Optional[User]:
    """Fetch user by email."""
    users = _db["users"]
    doc = await users.find_one({"email": email})
    
    if doc:
        return User(**_str_id(doc))
    return None


async def get_user_raw_by_email(email: str) -> Optional[dict]:
    """Fetch raw user document by email (no Pydantic validation)."""
    users = _db["users"]
    doc = await users.find_one({"email": email})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


async def get_all_users(limit: int = 100, skip: int = 0) -> List[UserProfile]:
    """Get all active users (admin only)."""
    users = _db["users"]
    docs = await users.find({"is_active": True}).skip(skip).limit(limit).to_list(None)
    return [UserProfile(**_str_id(doc)) for doc in docs]


async def update_user(user_id: str, update_data: dict) -> Optional[User]:
    """Update user fields."""
    users = _db["users"]
    update_data["updated_at"] = datetime.utcnow()
    
    try:
        result = await users.find_one_and_update(
            {"_id": ObjectId(user_id)},
            {"$set": update_data},
            return_document=True
        )
        if result:
            return User(**_str_id(result))
    except:
        pass
    return None


async def add_favorite_ticket(user_id: str, ticket_id: str) -> Optional[User]:
    """Add ticket to user's favorites."""
    users = _db["users"]
    try:
        result = await users.find_one_and_update(
            {"_id": ObjectId(user_id)},
            {"$addToSet": {"favorite_tickets": ticket_id}, "$set": {"updated_at": datetime.utcnow()}},
            return_document=True
        )
        if result:
            return User(**_str_id(result))
    except:
        pass
    return None


async def remove_favorite_ticket(user_id: str, ticket_id: str) -> Optional[User]:
    """Remove ticket from user's favorites."""
    users = _db["users"]
    try:
        result = await users.find_one_and_update(
            {"_id": ObjectId(user_id)},
            {"$pull": {"favorite_tickets": ticket_id}, "$set": {"updated_at": datetime.utcnow()}},
            return_document=True
        )
        if result:
            return User(**_str_id(result))
    except:
        pass
    return None


# ─────────────────────────────────────────────
# ANALYSIS HISTORY OPERATIONS
# ─────────────────────────────────────────────

async def save_analysis(
    user_id: str,
    history_data: AnalysisHistoryCreate
) -> AnalysisHistory:
    """Save a search/query to analysis history."""
    history = _db["analysis_history"]
    
    doc = {
        "user_id": user_id,
        **history_data.dict(),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    result = await history.insert_one(doc)
    doc["_id"] = result.inserted_id
    
    return AnalysisHistory(**_str_id(doc))


async def get_user_analysis_history(
    user_id: str,
    limit: int = 50,
    skip: int = 0
) -> tuple[List[AnalysisHistory], int]:
    """
    Get user's search/query history.
    Returns: (results, total_count)
    """
    history = _db["analysis_history"]
    
    # Get total count
    total_count = await history.count_documents({"user_id": user_id})
    print(f"[DEBUG DB] Querying analysis_history with user_id={user_id}, found {total_count} documents")
    
    # Get paginated results
    docs = await history.find({"user_id": user_id}) \
        .sort("created_at", DESCENDING) \
        .skip(skip) \
        .limit(limit) \
        .to_list(None)
    
    print(f"[DEBUG DB] Returning {len(docs)} documents for user {user_id}")
    return [AnalysisHistory(**_str_id(doc)) for doc in docs], total_count


async def update_analysis_entry(
    entry_id: str,
    user_id: str,
    update_data: dict
) -> Optional[AnalysisHistory]:
    """Update analysis entry (e.g., rating, tags)."""
    history = _db["analysis_history"]
    update_data["updated_at"] = datetime.utcnow()
    
    try:
        result = await history.find_one_and_update(
            {"_id": ObjectId(entry_id), "user_id": user_id},
            {"$set": update_data},
            return_document=True
        )
        if result:
            return AnalysisHistory(**_str_id(result))
    except:
        pass
    return None


async def delete_analysis_entry(entry_id: str, user_id: str) -> bool:
    """Delete a history entry (user can only delete their own)."""
    history = _db["analysis_history"]
    try:
        result = await history.delete_one({"_id": ObjectId(entry_id), "user_id": user_id})
        return result.deleted_count > 0
    except:
        return False


async def get_team_analysis_stats(team: str, days: int = 7) -> dict:
    """Get analysis statistics for a team (manager view)."""
    history = _db["analysis_history"]
    
    # Aggregate queries by team members
    pipeline = [
        {
            "$lookup": {
                "from": "users",
                "localField": "user_id",
                "foreignField": "_id",
                "as": "user"
            }
        },
        {
            "$match": {
                "user.team": team,
                "created_at": {"$gte": datetime.utcnow() - __import__("datetime").timedelta(days=days)}
            }
        },
        {
            "$group": {
                "_id": "$user_id",
                "query_count": {"$sum": 1},
                "avg_results": {"$avg": "$total_results"},
                "avg_rating": {"$avg": "$user_rating"}
            }
        }
    ]
    
    results = await history.aggregate(pipeline).to_list(None)
    return {"team": team, "stats": results, "period_days": days}


async def clear_all_tickets() -> int:
    """Delete all tickets from database."""
    tickets = _db["tickets"]
    result = await tickets.delete_many({})
    return result.deleted_count


# ─────────────────────────────────────────────
# PIPELINE JOB OPERATIONS
# ─────────────────────────────────────────────

async def create_pipeline_job(data: PipelineJobCreate) -> str:
    """Insert a new pipeline job record. Returns the job_id."""
    col = _db["pipeline_jobs"]
    now = datetime.utcnow()
    doc = {
        **data.dict(),
        "status": "processing",
        "progress": 0,
        "tickets_added": 0,
        "logs": [],
        "started_at": now,
        "finished_at": None,
        "error_message": None,
    }
    await col.insert_one(doc)
    return data.job_id


async def update_pipeline_job(job_id: str, **fields) -> None:
    """Patch arbitrary fields on a pipeline job."""
    col = _db["pipeline_jobs"]
    await col.update_one({"job_id": job_id}, {"$set": fields})


async def append_pipeline_log(job_id: str, message: str, level: str = "info") -> None:
    """Append a log entry to a pipeline job."""
    col = _db["pipeline_jobs"]
    entry = {"timestamp": datetime.utcnow(), "message": message, "level": level}
    await col.update_one({"job_id": job_id}, {"$push": {"logs": entry}})


async def get_pipeline_jobs(limit: int = 50) -> list:
    """Return most recent pipeline jobs (newest first)."""
    col = _db["pipeline_jobs"]
    cursor = col.find({}).sort("started_at", DESCENDING).limit(limit)
    docs = await cursor.to_list(length=limit)
    return [_str_id(d) for d in docs]


async def get_pipeline_job(job_id: str) -> Optional[dict]:
    """Return a single pipeline job by job_id."""
    col = _db["pipeline_jobs"]
    doc = await col.find_one({"job_id": job_id})
    return _str_id(doc) if doc else None


# ─────────────────────────────────────────────
# EVALUATION RUN OPERATIONS
# ─────────────────────────────────────────────

async def create_eval_run(data: EvalRunCreate) -> str:
    """Insert a new evaluation run record. Returns the run_id."""
    col = _db["eval_runs"]
    doc = {
        **data.dict(),
        "status": "processing",
        "progress": 0,
        "metrics": None,
        "by_team": None,
        "results": [],
        "logs": [],
        "started_at": datetime.utcnow(),
        "finished_at": None,
        "error_message": None,
    }
    await col.insert_one(doc)
    return data.run_id


async def update_eval_run(run_id: str, **fields) -> None:
    """Update arbitrary fields on an eval run."""
    col = _db["eval_runs"]
    await col.update_one({"run_id": run_id}, {"$set": fields})


async def append_eval_log(run_id: str, message: str, level: str = "info") -> None:
    """Append a timestamped log entry to an eval run."""
    col = _db["eval_runs"]
    entry = {
        "timestamp": datetime.utcnow(),
        "message": message,
        "level": level,
    }
    await col.update_one({"run_id": run_id}, {"$push": {"logs": entry}})


async def get_eval_runs(limit: int = 50) -> list:
    """Return most recent evaluation runs (newest first)."""
    col = _db["eval_runs"]
    cursor = col.find({}).sort("started_at", DESCENDING).limit(limit)
    docs = await cursor.to_list(length=limit)
    return [_str_id(d) for d in docs]


async def get_eval_run(run_id: str) -> Optional[dict]:
    """Return a single eval run by run_id."""
    col = _db["eval_runs"]
    doc = await col.find_one({"run_id": run_id})
    return _str_id(doc) if doc else None


# ─────────────────────────────────────────────
# AUDIT LOG OPERATIONS
# ─────────────────────────────────────────────

async def create_audit_log(data: AuditLogCreate) -> str:
    """Insert an audit log entry. Returns the inserted id."""
    col = _db["audit_logs"]
    doc = {
        **data.dict(),
        "created_at": datetime.utcnow(),
    }
    result = await col.insert_one(doc)
    return str(result.inserted_id)


async def get_audit_logs(
    limit: int = 100,
    skip: int = 0,
    action: Optional[str] = None,
    actor_email: Optional[str] = None,
) -> List[dict]:
    """Retrieve audit logs with optional filters, newest first."""
    col = _db["audit_logs"]
    query = {}
    if action:
        query["action"] = action
    if actor_email:
        query["actor_email"] = actor_email
    docs = await col.find(query).sort("created_at", DESCENDING).skip(skip).limit(limit).to_list(None)
    return [_str_id(d) for d in docs]


async def count_audit_logs(action: Optional[str] = None, actor_email: Optional[str] = None) -> int:
    """Count audit log entries with optional filters."""
    col = _db["audit_logs"]
    query = {}
    if action:
        query["action"] = action
    if actor_email:
        query["actor_email"] = actor_email
    return await col.count_documents(query)


# ─────────────────────────────────────────────
# NOTIFICATION OPERATIONS
# ─────────────────────────────────────────────

async def create_notification(data: NotificationCreate) -> str:
    """Insert a notification. Returns the inserted _id as str."""
    col = _db["notifications"]
    doc = {
        **data.dict(),
        "read": False,
        "created_at": datetime.utcnow(),
    }
    result = await col.insert_one(doc)
    return str(result.inserted_id)


async def create_notifications_for_admins(
    db_handle,
    notif_type: str,
    category: str,
    title: str,
    message: str,
    target_tab: Optional[str] = None,
) -> None:
    """Write one notification per ADMIN user in the DB."""
    users_col = db_handle["users"]
    notif_col = db_handle["notifications"]
    now = datetime.utcnow()
    admin_docs = await users_col.find({"role": "ADMIN", "is_active": True}).to_list(None)
    if not admin_docs:
        return
    docs = [
        {
            "user_id": str(u["_id"]),
            "type": notif_type,
            "category": category,
            "title": title,
            "message": message,
            "target_tab": target_tab,
            "read": False,
            "created_at": now,
        }
        for u in admin_docs
    ]
    await notif_col.insert_many(docs)


async def create_notifications_for_role_team(
    db_handle,
    roles: List[str],
    team: Optional[str],
    notif_type: str,
    category: str,
    title: str,
    message: str,
    target_tab: Optional[str] = None,
) -> None:
    """Write one notification per user matching given role(s) and optional team.
    If team is None, matches all teams (used for MANAGER who sees all)."""
    users_col = db_handle["users"]
    notif_col = db_handle["notifications"]
    now = datetime.utcnow()
    query: dict = {"role": {"$in": roles}, "is_active": True}
    if team is not None:
        query["team"] = team
    target_docs = await users_col.find(query).to_list(None)
    if not target_docs:
        return
    docs = [
        {
            "user_id": str(u["_id"]),
            "type": notif_type,
            "category": category,
            "title": title,
            "message": message,
            "target_tab": target_tab,
            "read": False,
            "created_at": now,
        }
        for u in target_docs
    ]
    await notif_col.insert_many(docs)


async def get_notifications(
    user_id: str,
    limit: int = 50,
    unread_only: bool = False,
) -> List[dict]:
    """Return notifications for a user, newest first."""
    col = _db["notifications"]
    query: dict = {"user_id": user_id}
    if unread_only:
        query["read"] = False
    docs = await col.find(query).sort("created_at", DESCENDING).limit(limit).to_list(None)
    return [_str_id(d) for d in docs]


async def count_unread_notifications(user_id: str) -> int:
    """Count unread notifications for a user."""
    col = _db["notifications"]
    return await col.count_documents({"user_id": user_id, "read": False})


async def mark_notification_read(notif_id: str, user_id: str) -> bool:
    """Mark a single notification as read. Returns True if updated."""
    col = _db["notifications"]
    try:
        result = await col.update_one(
            {"_id": ObjectId(notif_id), "user_id": user_id},
            {"$set": {"read": True}},
        )
        return result.modified_count > 0
    except Exception:
        return False


async def mark_all_notifications_read(user_id: str) -> int:
    """Mark all notifications as read for a user. Returns count updated."""
    col = _db["notifications"]
    result = await col.update_many(
        {"user_id": user_id, "read": False},
        {"$set": {"read": True}},
    )
    return result.modified_count
