#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
api.py
======
FastAPI HTTP server exposing the SopraHR RAG agent.

Start with:
  uvicorn scripts.api:app --reload --port 8080
  # or from the RAG-SopraHR/ root:
  uvicorn scripts.api:app --reload --host 0.0.0.0 --port 8080

Endpoints
---------
  GET  /health        → {"status": "ok"}
  POST /query         → SopraHROutput JSON
"""

import sys
import json
import secrets
import subprocess
import re
import os
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, ValidationError
from dotenv import load_dotenv
from jose import JWTError, jwt
from passlib.context import CryptContext

load_dotenv()

# ─────────────────────────────────────────────
# AUTH CONFIG
# ─────────────────────────────────────────────

_JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
_JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
_JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "8"))

# Use PBKDF2 to avoid bcrypt's 72-byte input constraint and backend quirks.
_pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
_bearer = HTTPBearer()

# Flat JSON file user store: data/users.json
# Schema: { "email": { "name": str, "hashed_password": str, "team": str } }
_USERS_FILE = Path(__file__).parent.parent / "data" / "users.json"

def _load_users() -> dict:
    if _USERS_FILE.exists():
        return json.loads(_USERS_FILE.read_text(encoding="utf-8"))
    return {}

def _save_users(users: dict) -> None:
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")

def _create_token(email: str, name: str, role: str = "CONSULTANT") -> str:
    payload = {
        "sub": email,
        "name": name,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=_JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)

def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide ou expiré")


def _require_role(*allowed_roles: str):
    """Dependency factory that checks the JWT role against allowed roles."""
    def checker(user: dict = Depends(_verify_token)) -> dict:
        user_role = user.get("role", "CONSULTANT")
        if user_role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Accès refusé: rôle insuffisant")
        return user
    return checker


_SCRIPTS = Path(__file__).parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from agent import create_rag_agent, run_query

# MongoDB Database & Routes
from scripts.database import init_db, close_db, get_user_by_email, get_user_raw_by_email, save_analysis as db_save_analysis, bulk_insert_tickets, get_ticket_by_reference, create_pipeline_job, update_pipeline_job, append_pipeline_log, get_pipeline_jobs as db_get_pipeline_jobs, create_eval_run, update_eval_run, append_eval_log, get_eval_runs as db_get_eval_runs, get_eval_run as db_get_eval_run, create_user as db_create_user, get_all_users as db_get_all_users, update_user as db_update_user, create_audit_log, get_audit_logs, count_audit_logs, create_notifications_for_admins, create_notifications_for_role_team, get_notifications, count_unread_notifications, mark_notification_read, mark_all_notifications_read
from scripts.models import AnalysisHistoryCreate, MatchedTicket, TicketCreate, ConversationEntry, PipelineJobCreate, EvalRunCreate, UserCreate, AuditLogCreate
from scripts.routes import router as db_router


def _push_admin_notification(notif_type: str, category: str, title: str, message: str, target_tab: str = None):
    """Fire-and-forget: write a notification for every ADMIN user in a background thread."""
    try:
        asyncio.run(_push_admin_notification_async(notif_type, category, title, message, target_tab))
    except Exception as e:
        print(f"[NOTIF] Failed to push admin notification: {e}")


async def _push_admin_notification_async(notif_type: str, category: str, title: str, message: str, target_tab: str = None):
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    try:
        db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
        await create_notifications_for_admins(db, notif_type, category, title, message, target_tab)
    finally:
        client.close()


def _push_role_team_notification(
    roles: list,
    team,  # str or None (None = all teams)
    notif_type: str,
    category: str,
    title: str,
    message: str,
    target_tab: str = None,
):
    """Fire-and-forget: write a notification for every user matching role(s)/team."""
    try:
        asyncio.run(_push_role_team_notification_async(roles, team, notif_type, category, title, message, target_tab))
    except Exception as e:
        print(f"[NOTIF] Failed to push role/team notification: {e}")


async def _push_role_team_notification_async(roles, team, notif_type, category, title, message, target_tab):
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    try:
        db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
        await create_notifications_for_role_team(db, roles, team, notif_type, category, title, message, target_tab)
    finally:
        client.close()


# ─────────────────────────────────────────────
# WEEKLY DIGEST SCHEDULER
# ─────────────────────────────────────────────

_scheduler = None


def _start_weekly_digest_scheduler():
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        _scheduler = BackgroundScheduler(timezone="Europe/Paris")
        # Every Monday at 08:00
        _scheduler.add_job(_run_weekly_digest_sync, CronTrigger(day_of_week="mon", hour=8, minute=0))
        _scheduler.start()
        print("[OK] Weekly digest scheduler started (every Monday 08:00 Paris)")
    except Exception as e:
        print(f"[WARN] Could not start weekly digest scheduler: {e}")


def _stop_weekly_digest_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)


def _run_weekly_digest_sync():
    """Synchronous wrapper called by APScheduler; runs the async digest in a new event loop."""
    try:
        asyncio.run(_send_weekly_digest())
    except Exception as e:
        print(f"[WEEKLY DIGEST] Error: {e}")


async def _send_weekly_digest():
    """Compute per-team stats for the past 7 days and notify TEAM_LEAD + MANAGER."""
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    try:
        db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
        teams = ("DSN", "Appli", "Outils")
        seven_days_ago = datetime.utcnow() - __import__("datetime").timedelta(days=7)

        for team in teams:
            # Get all user_ids in this team
            team_users = await db["users"].find({"team": team}).to_list(None)
            team_user_ids = [str(u["_id"]) for u in team_users]
            if not team_user_ids:
                continue

            # Aggregate stats for the week
            total = await db["analysis_history"].count_documents({
                "user_id": {"$in": team_user_ids},
                "created_at": {"$gte": seven_days_ago},
            })
            utile = await db["analysis_history"].count_documents({
                "user_id": {"$in": team_user_ids},
                "user_rating": {"$gte": 4},
                "created_at": {"$gte": seven_days_ago},
            })
            inutile = await db["analysis_history"].count_documents({
                "user_id": {"$in": team_user_ids},
                "user_rating": {"$lte": 2},
                "created_at": {"$gte": seven_days_ago},
            })
            unrated = await db["analysis_history"].count_documents({
                "user_id": {"$in": team_user_ids},
                "user_rating": None,
                "created_at": {"$gte": seven_days_ago},
            })

            helpful_pct = round(utile / total * 100) if total else 0
            msg = (
                f"Bilan semaine – équipe {team} : {total} analyse(s), "
                f"{helpful_pct}% utiles ({utile}↑ / {inutile}↓ / {unrated} non notées)."
            )

            # Notify TEAM_LEAD of this team
            await create_notifications_for_role_team(
                db, ["TEAM_LEAD"], team,
                "info", "analysis",
                f"Bilan hebdomadaire – {team}",
                msg,
                "history",
            )

        # Notify MANAGER with a cross-team summary (one notification, all teams)
        all_team_ids = [str(u["_id"]) for u in await db["users"].find({}).to_list(None)]
        total_all = await db["analysis_history"].count_documents({"created_at": {"$gte": seven_days_ago}})
        utile_all = await db["analysis_history"].count_documents({"user_rating": {"$gte": 4}, "created_at": {"$gte": seven_days_ago}})
        inutile_all = await db["analysis_history"].count_documents({"user_rating": {"$lte": 2}, "created_at": {"$gte": seven_days_ago}})
        helpful_all = round(utile_all / total_all * 100) if total_all else 0
        manager_msg = (
            f"Bilan toutes équipes : {total_all} analyse(s) la semaine passée, "
            f"{helpful_all}% utiles ({utile_all}↑ / {inutile_all}↓). "
            f"Détail par équipe disponible dans l'historique."
        )
        await create_notifications_for_role_team(
            db, ["MANAGER"], None,
            "info", "analysis",
            "Bilan hebdomadaire – Toutes équipes",
            manager_msg,
            "history",
        )
        print(f"[WEEKLY DIGEST] Sent for {len(teams)} teams, {total_all} total analyses")
    finally:
        client.close()


# ─────────────────────────────────────────────
# PATHS  (relative to project root)
# ─────────────────────────────────────────────

_ROOT      = _SCRIPTS.parent
_DB_DIR    = _ROOT / "data" / "indexes"
_BM25_DIR  = _ROOT / "data" / "bm25"
_INDEX_DIR = _ROOT / "data" / "pageindex"
_RAW_INBOX_DIR = _ROOT / "data" / "raw"
_SIMPLE_INBOX_DIR = _ROOT / "data" / "inbox" / "simple"
_JSON_DIR = _ROOT / "data" / "json"
_AUTO_PIPELINE = _SCRIPTS / "auto_pipeline.py"

VALID_TEAMS = ("DSN", "Appli", "Outils")

# ─────────────────────────────────────────────
# PIPELINE JOB TRACKER
# ─────────────────────────────────────────────

import threading
import asyncio

_pipeline_jobs: dict[str, dict] = {}
_pipeline_lock = threading.Lock()


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _save_uploaded_file(upload: UploadFile, mode: str) -> Path:
    target_dir = _RAW_INBOX_DIR if mode == "raw" else _SIMPLE_INBOX_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    original = _safe_name(upload.filename or "upload.txt")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = target_dir / f"{stamp}_{uuid4().hex[:8]}_{original}"

    data = upload.file.read()
    saved.write_bytes(data)
    return saved

# ─────────────────────────────────────────────
# PRE-LOAD AGENTS (one per team at startup)
# ─────────────────────────────────────────────

_agents: dict = {}
_main_loop: asyncio.AbstractEventLoop | None = None

def _get_agent(team: str):
    if team not in VALID_TEAMS:
        raise HTTPException(status_code=400, detail=f"team must be one of {VALID_TEAMS}")
    if team not in _agents:
        _agents[team] = create_rag_agent(
            team=team,
            db_dir=_DB_DIR,
            bm25_dir=_BM25_DIR,
            index_dir=_INDEX_DIR,
        )
    return _agents[team]


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(title="SopraHR RAG API", version="1.0.0")

# Include MongoDB routes
app.include_router(db_router, prefix="/api", tags=["tickets", "users", "analysis"])

# Allow requests from the Vite dev server and any local origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# LIFECYCLE EVENTS - MongoDB
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB connection on startup."""
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    try:
        await init_db()
        print("[OK] MongoDB initialized")
    except Exception as e:
        print(f"[ERROR] MongoDB startup failed: {e}")
        raise
    _start_weekly_digest_scheduler()


@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection on shutdown."""
    await close_db()
    _stop_weekly_digest_scheduler()


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    team: Literal["DSN", "Appli", "Outils"] = "DSN"


class PatchItem(BaseModel):
    patch: str = ""
    ref: str = ""


class TicketReference(BaseModel):
    ref: str = ""
    titre: str = ""
    resolution_ticket: str = ""
    patches: list[str] = []


class QueryResponse(BaseModel):
    analyse: str = ""
    tickets_utilises: list[str] = []
    cause_probable: str = ""
    resolution: str = ""
    tickets_references: list[TicketReference] = []
    patches: list[PatchItem] = []
    reponse_lotus: str = ""


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    team: Literal["DSN", "Appli", "Outils"] = "DSN"


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class DeleteAccountRequest(BaseModel):
    current_password: str


class AuthResponse(BaseModel):
    token: str
    name: str
    email: str
    team: str
    role: str


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────

@app.post("/auth/signup", response_model=AuthResponse)
async def signup(body: SignupRequest):
    existing = await get_user_raw_by_email(body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Un compte existe déjà avec cet e-mail")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Le mot de passe doit contenir au moins 8 caractères")

    password_hash = _pwd_ctx.hash(body.password)
    user_data = UserCreate(username=body.name, email=body.email, team=body.team, password=body.password)
    await db_create_user(user_data, password_hash, role="CONSULTANT")

    token = _create_token(body.email, body.name, "CONSULTANT")
    _log_audit_sync("signup", actor_email=body.email, actor_role="CONSULTANT",
                    details={"name": body.name, "team": body.team})
    return AuthResponse(token=token, name=body.name, email=body.email, team=body.team, role="CONSULTANT")


@app.post("/auth/login", response_model=AuthResponse)
async def login(body: LoginRequest):
    doc = await get_user_raw_by_email(body.email)

    if not doc:
        _log_audit_sync("login_failed", actor_email=body.email)
        raise HTTPException(status_code=401, detail="E-mail ou mot de passe incorrect")

    stored_hash = doc.get("password_hash") or doc.get("hashed_password", "")
    if not stored_hash or not _pwd_ctx.verify(body.password, stored_hash):
        _log_audit_sync("login_failed", actor_email=body.email)
        raise HTTPException(status_code=401, detail="E-mail ou mot de passe incorrect")

    name = doc.get("username") or doc.get("name") or body.email.split("@")[0]
    role = doc.get("role") or "CONSULTANT"
    team = doc.get("team")
    token = _create_token(body.email, name, role)
    _log_audit_sync("login", actor_email=body.email, actor_role=role)
    return AuthResponse(token=token, name=name, email=body.email, team=team, role=role)


@app.post("/auth/logout")
def logout(user: dict = Depends(_verify_token)):
    _log_audit_sync("logout", actor_email=user.get("sub", "unknown"),
                    actor_role=user.get("role", "CONSULTANT"))
    return {"status": "ok"}


@app.get("/auth/me")
def me(user: dict = Depends(_verify_token)):
    email = user["sub"]
    users = _load_users()
    user_data = users.get(email)
    
    if not user_data:
        return {
            "email": email,
            "name": email.split("@")[0],
            "team": "DSN",
            "role": user.get("role", "CONSULTANT"),
            "id": email
        }
    
    return {
        "email": email,
        "name": user_data.get("name", ""),
        "team": user_data.get("team", ""),
        "role": user_data.get("role", "CONSULTANT"),
        "id": email
    }


@app.post("/auth/change-password")
def change_password(body: ChangePasswordRequest, user: dict = Depends(_verify_token)):
    email = user.get("sub", "")
    users = _load_users()
    user_data = users.get(email)
    if not user_data:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    if not _pwd_ctx.verify(body.current_password, user_data.get("hashed_password", "")):
        raise HTTPException(status_code=400, detail="Mot de passe actuel incorrect")

    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="Le nouveau mot de passe doit contenir au moins 8 caractères")

    user_data["hashed_password"] = _pwd_ctx.hash(body.new_password)
    users[email] = user_data
    _save_users(users)
    _log_audit_sync("settings_changed", actor_email=email, actor_role=user.get("role", "CONSULTANT"),
                    details={"field": "password"})
    return {"status": "ok", "message": "Mot de passe mis à jour"}


@app.get("/auth/export-data")
async def export_my_data(user: dict = Depends(_verify_token)):
    """Export current user profile + analyses as JSON payload."""
    email = user.get("sub", "")
    users = _load_users()
    user_data = users.get(email, {})

    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    try:
        db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
        mongo_user = await db["users"].find_one({"email": email})
        analyses = []
        if mongo_user:
            user_id = str(mongo_user.get("_id"))
            docs = await db["analysis_history"].find({"user_id": user_id}).sort("created_at", -1).to_list(None)
            for d in docs:
                d["id"] = str(d.pop("_id"))
                analyses.append(d)
        payload = {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "user": {
                "email": email,
                "name": user_data.get("name", email.split("@")[0]),
                "team": user_data.get("team", ""),
                "role": user_data.get("role", user.get("role", "CONSULTANT")),
            },
            "analysis_history": analyses,
        }
    finally:
        client.close()

    _log_audit_sync("settings_changed", actor_email=email, actor_role=user.get("role", "CONSULTANT"),
                    details={"action": "export_data", "rows": len(payload.get("analysis_history", []))})
    return payload


@app.post("/auth/delete-account")
async def delete_my_account(body: DeleteAccountRequest, user: dict = Depends(_verify_token)):
    """Delete current account from auth store and MongoDB data tied to this user."""
    email = user.get("sub", "")
    users = _load_users()
    user_data = users.get(email)
    if not user_data:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    if not _pwd_ctx.verify(body.current_password, user_data.get("hashed_password", "")):
        raise HTTPException(status_code=400, detail="Mot de passe actuel incorrect")

    users.pop(email, None)
    _save_users(users)

    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    try:
        db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
        mongo_user = await db["users"].find_one({"email": email})
        if mongo_user:
            user_id = str(mongo_user.get("_id"))
            await db["analysis_history"].delete_many({"user_id": user_id})
            await db["notifications"].delete_many({"user_id": user_id})
        await db["users"].delete_many({"email": email})
    finally:
        client.close()

    _log_audit_sync("user_deleted", actor_email=email, actor_role=user.get("role", "CONSULTANT"),
                    target_type="user", details={"email": email})
    return {"status": "ok", "message": "Compte supprimé"}


# ─────────────────────────────────────────────
# ADMIN: USER MANAGEMENT
# ─────────────────────────────────────────────

VALID_ROLES = ("ADMIN", "MANAGER", "TEAM_LEAD", "CONSULTANT")


class AdminCreateUserRequest(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    team: Literal["DSN", "Appli", "Outils"] = "DSN"
    role: Literal["ADMIN", "MANAGER", "TEAM_LEAD", "CONSULTANT"] = "CONSULTANT"


class AdminUpdateUserRequest(BaseModel):
    name: Optional[str] = None
    team: Optional[Literal["DSN", "Appli", "Outils"]] = None
    role: Optional[Literal["ADMIN", "MANAGER", "TEAM_LEAD", "CONSULTANT"]] = None
    is_active: Optional[bool] = None


async def _log_audit(action: str, actor: dict, target_type: str = None, target_id: str = None, details: dict = None):
    """Helper to log an audit event from the current request user."""
    try:
        await create_audit_log(AuditLogCreate(
            action=action,
            actor_email=actor.get("sub", "unknown"),
            actor_role=actor.get("role", "CONSULTANT"),
            target_type=target_type,
            target_id=target_id,
            details=details,
        ))
    except Exception as e:
        print(f"[WARN] Audit log failed: {e}")


def _log_audit_sync(action: str, actor_email: str, actor_role: str = "CONSULTANT",
                    target_type: str = None, target_id: str = None, details: dict = None):
    """Fire-and-forget audit log from synchronous endpoints."""
    if _main_loop is None or _main_loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(
            create_audit_log(AuditLogCreate(
                action=action,
                actor_email=actor_email,
                actor_role=actor_role,
                target_type=target_type,
                target_id=target_id,
                details=details,
            )),
            _main_loop,
        )
    except Exception as e:
        print(f"[WARN] Sync audit log failed: {e}")


@app.get("/admin/users")
async def admin_list_users(user: dict = Depends(_require_role("ADMIN", "MANAGER"))):
    """List all users. ADMIN/MANAGER only."""
    db_users = await db_get_all_users(limit=500)
    return [
        {
            "id": u.id,
            "email": u.email,
            "name": u.username,
            "team": u.team or "",
            "role": u.role,
            "is_active": u.is_active,
        }
        for u in db_users
    ]


@app.post("/admin/users")
async def admin_create_user(body: AdminCreateUserRequest, user: dict = Depends(_require_role("ADMIN"))):
    """Create a new user in MongoDB. ADMIN only."""
    existing = await get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Un compte existe déjà avec cet e-mail")
    password_hash = _pwd_ctx.hash(body.password)
    try:
        user_data = UserCreate(username=body.name, email=body.email, password=body.password, team=body.team)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    new_user = await db_create_user(user_data, password_hash, role=body.role)
    await _log_audit("user_created", user, "user", body.email, {"role": body.role, "team": body.team})
    return {"id": new_user.id, "email": new_user.email, "name": new_user.username, "team": new_user.team or "", "role": new_user.role, "is_active": new_user.is_active}


@app.put("/admin/users/{email}")
async def admin_update_user(email: str, body: AdminUpdateUserRequest, user: dict = Depends(_require_role("ADMIN"))):
    """Update a user's name, team, role, or active status. ADMIN only."""
    db_user = await get_user_by_email(email)
    if not db_user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    changes = {}
    update_fields = {}
    if body.name is not None:
        update_fields["username"] = body.name
        changes["name"] = body.name
    if body.team is not None:
        update_fields["team"] = body.team
        changes["team"] = body.team
    if body.role is not None:
        changes["role"] = {"from": db_user.role, "to": body.role}
        update_fields["role"] = body.role
    if body.is_active is not None:
        update_fields["is_active"] = body.is_active
        changes["is_active"] = body.is_active
    if update_fields:
        updated = await db_update_user(db_user.id, update_fields)
        if not updated:
            raise HTTPException(status_code=500, detail="Échec de la mise à jour")
    else:
        updated = db_user
    await _log_audit("user_updated", user, "user", email, changes)
    return {
        "id": updated.id,
        "email": updated.email,
        "name": updated.username,
        "team": updated.team or "",
        "role": updated.role,
        "is_active": updated.is_active,
    }


@app.delete("/admin/users/{email}")
async def admin_delete_user(email: str, user: dict = Depends(_require_role("ADMIN"))):
    """Deactivate a user (soft-delete). ADMIN only."""
    db_user = await get_user_by_email(email)
    if not db_user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    if email == user.get("sub"):
        raise HTTPException(status_code=400, detail="Impossible de supprimer votre propre compte")
    await db_update_user(db_user.id, {"is_active": False})
    await _log_audit("user_deleted", user, "user", email)
    return {"status": "ok", "message": f"Utilisateur {email} désactivé"}


# ─────────────────────────────────────────────
# AUDIT LOG ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/admin/audit-logs")
async def get_audit_logs_endpoint(
    limit: int = 100,
    skip: int = 0,
    action: Optional[str] = None,
    actor: Optional[str] = None,
    user: dict = Depends(_require_role("ADMIN")),
):
    """Retrieve audit logs. ADMIN only."""
    logs = await get_audit_logs(limit=limit, skip=skip, action=action, actor_email=actor)
    total = await count_audit_logs(action=action, actor_email=actor)
    return {"logs": logs, "total": total}


# ─────────────────────────────────────────────
# PROTECTED ROUTES
# ─────────────────────────────────────────────

async def _save_query_history(user_email: str, question: str, result: dict, team: str, execution_time_ms: int) -> None:
    """Background task: persist a RAG query to MongoDB analysis history."""
    try:
        user = await get_user_by_email(user_email)
        if not user:
            return
        matched = [
            MatchedTicket(
                ticket_id="",
                reference=ref,
                title=ref,
                relevance_score=1.0,
                support_team=team,
            )
            for ref in (result.get("tickets_utilises") or [])
        ]
        # Build a composite answer from all non-empty text fields
        text_parts = [
            result.get("analyse", ""),
            result.get("cause_probable", ""),
            result.get("resolution", ""),
            result.get("reponse_lotus", ""),
        ]
        generated_answer = "\n\n".join(p for p in text_parts if p.strip()) or None

        history_data = AnalysisHistoryCreate(
            query=question,
            query_type="rag_query",
            matched_tickets=matched,
            generated_answer=generated_answer,
            execution_time_ms=execution_time_ms,
            total_results=len(matched),
            filters_applied={
                "team": team,
                "full_result": {
                    k: v for k, v in result.items()
                    if k not in ("tickets_utilises",) and v
                },
            },
        )
        await db_save_analysis(str(user.id), history_data)

        # Notify the consultant to rate their analysis
        from motor.motor_asyncio import AsyncIOMotorClient as _AMC
        _c = _AMC(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
        try:
            _db2 = _c[os.getenv("MONGODB_DB", "soprahr_rag")]
            await _db2["notifications"].insert_one({
                "user_id": str(user.id),
                "type": "info",
                "category": "analysis",
                "title": "Notez votre analyse",
                "message": f'Pensez à évaluer la qualité de l\'analyse pour : "{question[:120]}".',
                "target_tab": "history",
                "read": False,
                "created_at": datetime.utcnow(),
            })
        finally:
            _c.close()
    except Exception as exc:
        import traceback
        print(f"[WARN] Failed to save query history: {exc}\n{traceback.format_exc()}")


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, background_tasks: BackgroundTasks, _user: dict = Depends(_verify_token)):
    if body.team not in VALID_TEAMS:
        raise HTTPException(status_code=400, detail=f"team must be one of {VALID_TEAMS}")
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    agent = _get_agent(body.team)
    t0 = datetime.now()
    result = run_query(agent, body.question)
    exec_ms = int((datetime.now() - t0).total_seconds() * 1000)

    print(
        f"[QUERY] team={body.team} exec_ms={exec_ms} "
        f"tickets_utilises={result.get('tickets_utilises')} "
        f"tickets_references_count={len(result.get('tickets_references') or [])} "
        f"error={result.get('error')}"
    )

    # Only save successful analyses (not failed ones with errors)
    # Success criteria: no error field AND at least one content field has data
    has_error = result.get("error") is not None

    def _as_str(val, default: str = "") -> str:
        if val is None:
            return default
        if isinstance(val, list):
            return " ".join(str(v) for v in val if v)
        if isinstance(val, dict):
            return " ".join(str(v) for v in val.values() if v)
        return str(val)

    has_content = any([
        _as_str(result.get("analyse")).strip(),
        _as_str(result.get("cause_probable")).strip(),
        _as_str(result.get("resolution")).strip(),
        _as_str(result.get("reponse_lotus")).strip(),
        result.get("tickets_utilises", []),
    ])
    
    if not has_error and has_content:
        # Auto-save only successful analyses
        background_tasks.add_task(
            _save_query_history,
            _user.get("sub", ""),
            body.question,
            result if isinstance(result, dict) else result,
            body.team,
            exec_ms,
        )
    else:
        print(f"[DEBUG] Skipped saving analysis: has_error={has_error}, has_content={has_content}")

    _log_audit_sync(
        "query_executed" if not has_error else "query_failed",
        actor_email=_user.get("sub", "unknown"),
        actor_role=_user.get("role", "CONSULTANT"),
        target_type="analysis",
        details={"question": body.question[:200], "team": body.team, "exec_ms": exec_ms, "success": not has_error},
    )

    # Notify team lead of that team + all managers on failure
    if has_error:
        actor = _user.get("sub", "inconnu")
        err_msg = str(result.get("error", ""))[:200]
        threading.Thread(
            target=_push_role_team_notification,
            args=(["TEAM_LEAD"], body.team, "alert", "analysis",
                  f"Échec d'analyse ({body.team})",
                  f'L\'analyse de {actor} a échoué sur : "{body.question[:100]}". {err_msg}',
                  "history"),
            daemon=True,
        ).start()
        threading.Thread(
            target=_push_role_team_notification,
            args=(["MANAGER"], None, "alert", "analysis",
                  f"Échec d'analyse ({body.team})",
                  f'L\'analyse de {actor} ({body.team}) a échoué : "{body.question[:100]}". {err_msg}',
                  "history"),
            daemon=True,
        ).start()

    return result


@app.post("/pipeline/ingest")
async def pipeline_ingest(
    file: UploadFile = File(...),
    mode: Literal["raw", "simple"] = Form("simple"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _user: dict = Depends(_require_role("ADMIN")),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant")

    saved_path = _save_uploaded_file(file, mode)
    job_id = f"PJ-{uuid4().hex[:12]}"
    user_email = _user.get("sub", "unknown")

    with _pipeline_lock:
        _pipeline_jobs[job_id] = {
            "id": job_id,
            "file": saved_path.name,
            "mode": mode,
            "status": "processing",
            "progress": 10,
            "message": "Fichier sauvegardé, démarrage du pipeline...",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "tickets_added": 0,
        }

    # Persist to MongoDB
    try:
        await create_pipeline_job(PipelineJobCreate(
            job_id=job_id,
            file_name=saved_path.name,
            mode=mode,
            triggered_by=user_email,
        ))
        await append_pipeline_log(job_id, f"Fichier reçu: {saved_path.name} (mode={mode})")
    except Exception as e:
        print(f"[PIPELINE] Failed to save job to DB: {e}")

    background_tasks.add_task(_run_pipeline_background, job_id, saved_path, mode)

    await _log_audit("pipeline_started", _user, target_type="pipeline", target_id=job_id,
                     details={"file": saved_path.name, "mode": mode})

    return {"status": "accepted", "job_id": job_id, "file": saved_path.name, "mode": mode}


def _update_job(job_id: str, **kwargs):
    with _pipeline_lock:
        if job_id in _pipeline_jobs:
            _pipeline_jobs[job_id].update(kwargs)


def _db_log(job_id: str, message: str, level: str = "info"):
    """Fire-and-forget DB log from a background thread."""
    try:
        asyncio.run(_db_log_async(job_id, message, level))
    except Exception as e:
        print(f"[PIPELINE DB LOG] Failed: {e}")


async def _db_log_async(job_id: str, message: str, level: str):
    from motor.motor_asyncio import AsyncIOMotorClient
    from datetime import datetime
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    try:
        db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
        entry = {"timestamp": datetime.utcnow(), "message": message, "level": level}
        await db["pipeline_jobs"].update_one({"job_id": job_id}, {"$push": {"logs": entry}})
    finally:
        client.close()


def _db_update_job(job_id: str, **fields):
    """Fire-and-forget DB update from a background thread."""
    try:
        asyncio.run(_db_update_job_async(job_id, **fields))
    except Exception as e:
        print(f"[PIPELINE DB UPDATE] Failed: {e}")


async def _db_update_job_async(job_id: str, **fields):
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    try:
        db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
        await db["pipeline_jobs"].update_one({"job_id": job_id}, {"$set": fields})
    finally:
        client.close()


def _run_pipeline_background(job_id: str, saved_path: Path, mode: str):
    """Run the auto_pipeline subprocess, then save parsed tickets to MongoDB."""
    try:
        pipeline_started_ts = datetime.now().timestamp()
        _update_job(job_id, progress=20, message="Exécution du pipeline d'ingestion...")
        _db_log(job_id, "Démarrage de l'exécution du pipeline...")
        print(f"[PIPELINE {job_id}] Starting: file={saved_path}, mode={mode}")

        cmd = [sys.executable, str(_AUTO_PIPELINE)]
        if mode == "raw":
            cmd += ["--raw-file", str(saved_path)]
        else:
            cmd += ["--simple-file", str(saved_path)]

        run = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        output = (run.stdout or "") + ("\n" + run.stderr if run.stderr else "")
        print(f"[PIPELINE {job_id}] Pipeline output:\n{output}")

        if run.returncode != 0:
            print(f"[PIPELINE {job_id}] Pipeline failed with return code {run.returncode}")
            _update_job(job_id, status="failed", progress=100, message=f"Échec pipeline: {output[-500:]}")
            _db_update_job(
                job_id,
                status="failed",
                progress=100,
                finished_at=datetime.utcnow(),
                error_message=output[-1000:],
            )
            _db_log(job_id, f"Échec pipeline (code {run.returncode}): {output[-300:]}", level="error")
            return

        _update_job(job_id, progress=70, message="Pipeline terminé, sauvegarde en base de données...")
        _db_log(job_id, "Pipeline terminé. Conversion et indexation achevées.")
        # Log a summary from the output
        for line in output.splitlines():
            if any(k in line for k in ("Created", "Updated", "Converted", "Saved", "Built", "finished")):
                _db_log(job_id, line.strip())
        print(f"[PIPELINE {job_id}] Pipeline output completed, now saving to DB...")

        # Save newly generated JSON tickets to MongoDB
        tickets_added = _save_new_tickets_to_db(saved_path, mode, pipeline_started_ts)
        print(f"[PIPELINE {job_id}] Saved {tickets_added} tickets to MongoDB")
        _db_log(job_id, f"{tickets_added} ticket(s) sauvegardé(s) en base de données.")

        _update_job(job_id, progress=90, message="Rechargement des agents RAG...", tickets_added=tickets_added)

        # Reload RAG agents so new indexes are picked up
        _reload_agents()
        _db_log(job_id, "Agents RAG rechargés avec les nouveaux index.")

        _update_job(
            job_id, status="completed", progress=100,
            message=f"Pipeline terminé. {tickets_added} ticket(s) ajouté(s) à la base.",
            output=output[-2000:],
        )
        _db_update_job(
            job_id,
            status="completed",
            progress=100,
            tickets_added=tickets_added,
            finished_at=datetime.utcnow(),
        )
        _db_log(job_id, f"Job terminé avec succès. {tickets_added} ticket(s) ajouté(s).")
        print(f"[PIPELINE {job_id}] Job completed successfully")
        _push_admin_notification(
            "success", "pipeline",
            "Pipeline terminé",
            f"Le job {job_id} ({saved_path.name}) s'est terminé avec succès. {tickets_added} ticket(s) ajouté(s).",
            "pipeline",
        )

    except Exception as exc:
        import traceback
        error_msg = f"{exc}\n{traceback.format_exc()}"
        print(f"[PIPELINE ERROR {job_id}] {error_msg}")
        _update_job(job_id, status="failed", progress=100, message=str(exc))
        _db_update_job(job_id, status="failed", progress=100, finished_at=datetime.utcnow(), error_message=str(exc))
        _db_log(job_id, f"Erreur inattendue: {exc}", level="error")
        _push_admin_notification(
            "alert", "pipeline",
            "Erreur pipeline",
            f"Le job {job_id} a échoué: {str(exc)[:200]}",
            "pipeline",
        )


def _save_new_tickets_to_db(saved_path: Path, mode: str, started_ts: float) -> int:
    """Read the JSON files created by the pipeline and save them to MongoDB."""
    json_dir = _JSON_DIR
    print(f"[PIPELINE SAVE] Looking for JSON files in: {json_dir}")
    
    if not json_dir.exists():
        print(f"[PIPELINE SAVE] JSON directory does not exist: {json_dir}")
        return 0

    # Select files touched during this pipeline run.
    # Long runs can exceed fixed windows (e.g. 30 min), so use the actual job start timestamp.
    cutoff = max(0.0, started_ts - 5)
    all_jsons = list(json_dir.glob("*.json"))
    print(f"[PIPELINE SAVE] Found {len(all_jsons)} total JSON files")
    
    new_jsons = [
        f for f in all_jsons
        if f.stat().st_mtime > cutoff
    ]
    print(f"[PIPELINE SAVE] Found {len(new_jsons)} JSON files changed since pipeline start")
    
    if not new_jsons:
        print(f"[PIPELINE SAVE] No new JSON files found (cutoff={cutoff}, now={datetime.now().timestamp()})")
        # List file timestamps for debugging
        for f in all_jsons[-5:]:  # Show the 5 most recent files
            mtime = f.stat().st_mtime
            age_sec = datetime.now().timestamp() - mtime
            print(f"  - {f.name}: mtime={mtime}, age={age_sec:.1f}s ago")
        return 0

    print(f"[PIPELINE SAVE] Processing {len(new_jsons)} new JSON files...")
    tickets_to_insert: list[TicketCreate] = []
    
    for jf in new_jsons:
        try:
            print(f"[PIPELINE SAVE] Parsing {jf.name}...")
            data = json.loads(jf.read_text(encoding="utf-8"))
            reference = data.get("reference", "")
            title = data.get("title", "")
            description = data.get("description", "")

            if not reference or not title:
                print(f"[PIPELINE SAVE]   Skipping {jf.name}: missing reference or title")
                continue

            print(f"[PIPELINE SAVE]   Creating ticket: {reference}")

            # Build conversation entries from raw JSON, mapping "content" → "text"
            raw_convs = data.get("conversation", [])
            conv_entries = []
            for c in raw_convs:
                body_text = c.get("text") or c.get("content") or None
                entry = ConversationEntry(
                    timestamp=str(c.get("timestamp", "")),
                    actor=c.get("actor", "client"),
                    author=c.get("author", ""),
                    type=c.get("type", "message"),
                    action=c.get("action") or None,
                    text=body_text,
                    content=None,
                )
                conv_entries.append(entry)

            ticket = TicketCreate(
                reference=reference,
                title=title,
                source_file=jf.name,
                version=data.get("version"),
                system=data.get("system"),
                site_env=data.get("site_env"),
                support_team=data.get("support_team"),
                team_source=data.get("team_source"),
                closing_teamcode=data.get("closing_teamcode"),
                closing_status_code=data.get("closing_status_code"),
                closing_status_explanation=data.get("closing_status_explanation"),
                closing_level=data.get("closing_level"),
                is_closed=data.get("closing_status_code", "").upper() in ("CP", "C", "CL"),
                description=description,
                resolution=data.get("resolution"),
                espdsn_version=data.get("espdsn_version"),
                patches=data.get("patches", []),
                conversation=conv_entries,
                message_count=len(conv_entries),
                confidence=data.get("confidence"),
                content_hash=data.get("content_hash"),
            )
            tickets_to_insert.append(ticket)
        except Exception as e:
            print(f"[PIPELINE SAVE] Failed to parse {jf.name}: {e}")

    print(f"[PIPELINE SAVE] Prepared {len(tickets_to_insert)} tickets for DB insertion")
    
    if not tickets_to_insert:
        return 0

    # Run the async DB insert in a new event loop (we're in a background thread)
    print(f"[PIPELINE SAVE] Starting async DB insert...")
    count = asyncio.run(_async_bulk_insert(tickets_to_insert))
    print(f"[PIPELINE SAVE] Async DB insert completed, added {count} tickets")
    return count


async def _async_bulk_insert(tickets: list[TicketCreate]) -> int:
    """Insert tickets, skipping duplicates by reference. Uses local Motor client (background thread safe)."""
    import motor.motor_asyncio as _motor

    _MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    _MONGO_DB  = os.getenv("MONGODB_DB", "soprahr_rag")

    print(f"[PIPELINE DB] Connecting with local Motor client...")
    client = _motor.AsyncIOMotorClient(_MONGO_URL)
    db = client[_MONGO_DB]
    try:
        new_tickets = []
        updated_count = 0
        now = datetime.now()

        # Fields to compare for changes (skip metadata fields)
        _COMPARE_FIELDS = (
            "title", "description", "resolution", "version", "system",
            "site_env", "support_team", "team_source", "closing_teamcode",
            "closing_status_code", "closing_status_explanation", "closing_level",
            "is_closed", "patches", "conversation", "message_count",
            "confidence", "content_hash",
        )

        def _serialize_ticket(t):
            d = t.model_dump()
            if d.get("conversation"):
                d["conversation"] = [
                    c.model_dump() if hasattr(c, "model_dump") else (c.dict() if hasattr(c, "dict") else c)
                    for c in d["conversation"]
                ]
            return d

        print(f"[PIPELINE DB] Checking {len(tickets)} tickets...")
        for t in tickets:
            existing = await db["tickets"].find_one({"reference": t.reference})
            if existing is None:
                new_tickets.append(t)
                print(f"[PIPELINE DB]   {t.reference} - NEW")
            else:
                # Compare fields to detect changes
                t_dict = _serialize_ticket(t)
                changes = {}
                for field in _COMPARE_FIELDS:
                    new_val = t_dict.get(field)
                    old_val = existing.get(field)
                    # Normalize for comparison: treat None and empty list/string as equivalent
                    if new_val is None and old_val in (None, "", []):
                        continue
                    if old_val is None and new_val in (None, "", []):
                        continue
                    if new_val != old_val:
                        # Special case: don't overwrite non-empty conversation with empty
                        if field == "conversation" and not new_val and old_val:
                            continue
                        changes[field] = new_val

                if changes:
                    changes["updated_at"] = now
                    await db["tickets"].update_one(
                        {"reference": t.reference},
                        {"$set": changes},
                    )
                    updated_count += 1
                    changed_keys = ", ".join(changes.keys() - {"updated_at"})
                    print(f"[PIPELINE DB]   {t.reference} - UPDATED ({changed_keys})")
                else:
                    print(f"[PIPELINE DB]   {t.reference} - UNCHANGED (skipping)")

        inserted_count = 0
        if new_tickets:
            docs = []
            for t in new_tickets:
                d = _serialize_ticket(t)
                d["created_at"] = now
                d["updated_at"] = now
                docs.append(d)

            print(f"[PIPELINE DB] Inserting {len(docs)} new tickets...")
            result = await db["tickets"].insert_many(docs)
            inserted_count = len(result.inserted_ids)

        total = inserted_count + updated_count
        print(f"[PIPELINE DB] Done: {inserted_count} inserted, {updated_count} updated")
        return total
    except Exception as e:
        print(f"[PIPELINE DB] Error during insertion: {e}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        print(f"[PIPELINE DB] Closing local Motor client...")
        client.close()


def _reload_agents():
    """Clear cached RAG agents so they reload with fresh indexes."""
    global _agents
    _agents.clear()
    print("[PIPELINE] RAG agents cleared — will reload on next query")


@app.get("/pipeline/jobs/{job_id}")
def pipeline_job_status(job_id: str, _user: dict = Depends(_require_role("ADMIN"))):
    with _pipeline_lock:
        job = _pipeline_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/pipeline/jobs")
def pipeline_jobs_list(_user: dict = Depends(_require_role("ADMIN"))):
    with _pipeline_lock:
        return list(_pipeline_jobs.values())


@app.get("/pipeline/history")
async def pipeline_history(limit: int = 50, _user: dict = Depends(_require_role("ADMIN"))):
    """Return full pipeline job history from MongoDB, including logs."""
    try:
        jobs = await db_get_pipeline_jobs(limit=limit)
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/stats")
async def pipeline_stats(_user: dict = Depends(_require_role("ADMIN"))):
    """Return real stats about the RAG knowledge base."""
    from scripts.database import init_db
    import pickle

    # Count tickets in MongoDB
    ticket_count = 0
    team_counts: dict[str, int] = {}
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        db = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))[os.getenv("MONGODB_DB", "soprahr_rag")]
        ticket_count = await db["tickets"].count_documents({})
        for team in VALID_TEAMS:
            team_counts[team] = await db["tickets"].count_documents({"support_team": team})
    except Exception as e:
        print(f"[STATS] DB count failed: {e}")

    # Count indexed tickets (from FAISS config files)
    indexed_count = 0
    for team in VALID_TEAMS:
        config_file = _DB_DIR / team.lower() / "config.json"
        if config_file.exists():
            try:
                cfg = json.loads(config_file.read_text())
                indexed_count += cfg.get("count", 0)
            except Exception:
                pass

    # Count JSON files in data/json
    json_count = len(list(_JSON_DIR.glob("*.json"))) if _JSON_DIR.exists() else 0

    # Last pipeline job info
    last_job = None
    with _pipeline_lock:
        completed = [j for j in _pipeline_jobs.values() if j.get("status") == "completed"]
        if completed:
            last_job = max(completed, key=lambda j: j.get("started_at", ""))

    return {
        "ticket_count": ticket_count,
        "team_counts": team_counts,
        "indexed_count": indexed_count,
        "json_count": json_count,
        "last_job": last_job,
    }


@app.post("/tickets/reload-conversations")
async def reload_conversations(_user: dict = Depends(_require_role("ADMIN"))):
    """
    Re-read all JSON files (data/json + data/cleaned) and backfill the
    conversation field for tickets that currently have an empty conversation.
    Returns counts of tickets updated and skipped.
    """
    from motor.motor_asyncio import AsyncIOMotorClient
    db = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))[
        os.getenv("MONGODB_DB", "soprahr_rag")
    ]
    _cleaned_dir = _ROOT / "data" / "cleaned"
    dirs_to_scan = [d for d in [_JSON_DIR, _cleaned_dir] if d and d.exists()]

    updated = 0
    skipped = 0

    for json_dir in dirs_to_scan:
        for jf in json_dir.glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                reference = data.get("reference", "")
                raw_convs = data.get("conversation", [])
                if not reference or not raw_convs:
                    continue

                # Only update tickets that still have an empty conversation
                existing = await db["tickets"].find_one(
                    {"reference": reference},
                    {"_id": 1, "conversation": 1}
                )
                if not existing:
                    continue
                if existing.get("conversation"):
                    skipped += 1
                    continue

                # Build conversation list
                conv_entries = []
                for c in raw_convs:
                    body_text = c.get("text") or c.get("content") or None
                    conv_entries.append({
                        "timestamp": str(c.get("timestamp", "")),
                        "actor": c.get("actor", "client"),
                        "author": c.get("author", ""),
                        "type": c.get("type", "message"),
                        "action": c.get("action") or None,
                        "text": body_text,
                        "content": None,
                    })

                await db["tickets"].update_one(
                    {"reference": reference},
                    {"$set": {"conversation": conv_entries, "message_count": len(conv_entries)}}
                )
                updated += 1
            except Exception as e:
                print(f"[RELOAD-CONV] Error on {jf.name}: {e}")

    await _log_audit("tickets_reloaded", _user, target_type="ticket",
                     details={"updated": updated, "skipped": skipped})

    return {"updated": updated, "skipped": skipped, "message": f"Conversations rechargées: {updated} tickets mis à jour, {skipped} déjà remplis."}


# ─────────────────────────────────────────────
# EVALUATION TRACKER
# ─────────────────────────────────────────────

_eval_jobs: dict[str, dict] = {}
_eval_lock = threading.Lock()

_TEST_FILE = _ROOT / "Test" / "test_queries.json"
_EVAL_SCRIPT = _SCRIPTS / "evaluate_agent.py"


def _update_eval(run_id: str, **kwargs):
    with _eval_lock:
        if run_id in _eval_jobs:
            _eval_jobs[run_id].update(kwargs)


def _eval_db_log(run_id: str, message: str, level: str = "info"):
    """Fire-and-forget DB log from a background thread."""
    try:
        asyncio.run(_eval_db_log_async(run_id, message, level))
    except Exception as e:
        print(f"[EVAL DB LOG] Failed: {e}")


async def _eval_db_log_async(run_id: str, message: str, level: str):
    from motor.motor_asyncio import AsyncIOMotorClient
    from datetime import datetime
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    try:
        db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
        entry = {"timestamp": datetime.utcnow(), "message": message, "level": level}
        await db["eval_runs"].update_one({"run_id": run_id}, {"$push": {"logs": entry}})
    finally:
        client.close()


def _eval_db_update(run_id: str, **fields):
    """Fire-and-forget DB update from a background thread."""
    try:
        asyncio.run(_eval_db_update_async(run_id, **fields))
    except Exception as e:
        print(f"[EVAL DB UPDATE] Failed: {e}")


async def _eval_db_update_async(run_id: str, **fields):
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    try:
        db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
        await db["eval_runs"].update_one({"run_id": run_id}, {"$set": fields})
    finally:
        client.close()


def _run_eval_background(run_id: str, eval_type: str, team_filter: Optional[str], samples: Optional[int]):
    """Run the evaluate_agent.py script as a subprocess and parse its JSON output."""
    try:
        _update_eval(run_id, progress=10, message="Chargement des questions de test...")
        _eval_db_log(run_id, f"Démarrage évaluation: type={eval_type}, team={team_filter or 'all'}, samples={samples or 'all'}")

        # Build command
        cmd = [sys.executable, str(_EVAL_SCRIPT)]
        if team_filter:
            cmd += ["--team", team_filter]
        else:
            cmd += ["--team", "all"]
        if samples:
            cmd += ["--samples", str(samples)]

        print(f"[EVAL {run_id}] Running: {' '.join(cmd)}")
        _update_eval(run_id, progress=20, message="Exécution de l'évaluation agent...")
        _eval_db_log(run_id, "Exécution du script d'évaluation...")

        run = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        output = (run.stdout or "") + ("\n" + run.stderr if run.stderr else "")
        print(f"[EVAL {run_id}] Script output:\n{output[:2000]}")

        if run.returncode != 0:
            _update_eval(run_id, status="failed", progress=100, message=f"Échec évaluation (code {run.returncode})")
            _eval_db_update(run_id, status="failed", progress=100, finished_at=datetime.utcnow(), error_message=output[-1000:])
            _eval_db_log(run_id, f"Échec (code {run.returncode}): {output[-300:]}", level="error")
            return

        _update_eval(run_id, progress=70, message="Lecture des résultats...")
        _eval_db_log(run_id, "Script terminé. Lecture des résultats JSON...")

        # Find the most recent eval output file
        output_dir = _ROOT / "output"
        eval_files = sorted(output_dir.glob("eval_agent_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not eval_files:
            _update_eval(run_id, status="failed", progress=100, message="Aucun fichier résultat trouvé")
            _eval_db_update(run_id, status="failed", progress=100, finished_at=datetime.utcnow(), error_message="No eval output file found")
            _eval_db_log(run_id, "Aucun fichier résultat trouvé dans output/", level="error")
            return

        result_file = eval_files[0]
        _eval_db_log(run_id, f"Fichier résultat: {result_file.name}")

        with open(result_file, "r", encoding="utf-8") as f:
            report = json.load(f)

        metrics = report.get("metrics", {})
        by_team = report.get("by_team", {})
        results = report.get("results", [])

        # Strip agent_output from results for DB storage (can be very large)
        results_slim = []
        for r in results:
            slim = {k: v for k, v in r.items() if k != "agent_output"}
            results_slim.append(slim)

        _update_eval(run_id, progress=90, message="Sauvegarde en base de données...")
        _eval_db_log(run_id, f"Métriques: {metrics.get('total_questions', 0)} questions, "
                     f"ref_hit={metrics.get('ref_hit_rate', 0):.1%}, "
                     f"effective={metrics.get('effective_rate', 0):.1%}, "
                     f"overlap={metrics.get('mean_overlap', 0):.2f}")

        # Log per-team summary
        for team_name, team_data in by_team.items():
            _eval_db_log(run_id, f"  {team_name}: {team_data.get('questions', 0)}q, "
                         f"ref_hit={team_data.get('ref_hit_rate', 0):.1%}, "
                         f"effective={team_data.get('effective_rate', 0):.1%}")

        _eval_db_update(
            run_id,
            status="completed",
            progress=100,
            metrics=metrics,
            by_team=by_team,
            results=results_slim,
            finished_at=datetime.utcnow(),
        )

        n = metrics.get("total_questions", 0)
        eff = metrics.get("effective_rate", 0)
        _update_eval(
            run_id,
            status="completed",
            progress=100,
            message=f"Évaluation terminée. {n} questions, taux effectif: {eff:.1%}",
            metrics=metrics,
            by_team=by_team,
        )
        _eval_db_log(run_id, f"Évaluation terminée avec succès. {n} questions évaluées.")
        print(f"[EVAL {run_id}] Completed: {n} questions, effective_rate={eff:.1%}")
        _push_admin_notification(
            "success", "evaluation",
            "Evaluation terminée",
            f"Le run {run_id} est terminé: {n} questions, taux effectif {eff:.1%}.",
            "evaluations",
        )

    except Exception as exc:
        import traceback
        error_msg = f"{exc}\n{traceback.format_exc()}"
        print(f"[EVAL ERROR {run_id}] {error_msg}")
        _update_eval(run_id, status="failed", progress=100, message=str(exc))
        _eval_db_update(run_id, status="failed", progress=100, finished_at=datetime.utcnow(), error_message=str(exc))
        _eval_db_log(run_id, f"Erreur inattendue: {exc}", level="error")
        _push_admin_notification(
            "alert", "evaluation",
            "Erreur evaluation",
            f"Le run {run_id} a échoué: {str(exc)[:200]}",
            "evaluations",
        )


# ─────────────────────────────────────────────
# EVALUATION ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/eval/start")
async def eval_start(
    background_tasks: BackgroundTasks,
    eval_type: str = "agent",
    team: Optional[str] = None,
    samples: Optional[int] = None,
    _user: dict = Depends(_require_role("ADMIN")),
):
    """Start an evaluation run as a background task."""
    if eval_type not in ("agent", "hybrid", "vectorless"):
        raise HTTPException(status_code=400, detail=f"Type d'évaluation invalide: {eval_type}")
    if team and team.lower() != "all" and team not in VALID_TEAMS:
        raise HTTPException(status_code=400, detail=f"Équipe invalide: {team}")

    run_id = f"EV-{uuid4().hex[:12]}"
    user_email = _user.get("sub", "unknown")
    team_filter = None if (not team or team.lower() == "all") else team

    with _eval_lock:
        _eval_jobs[run_id] = {
            "id": run_id,
            "eval_type": eval_type,
            "team_filter": team_filter,
            "samples": samples,
            "status": "processing",
            "progress": 5,
            "message": "Initialisation de l'évaluation...",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "metrics": None,
            "by_team": None,
        }

    try:
        await create_eval_run(EvalRunCreate(
            run_id=run_id,
            eval_type=eval_type,
            team_filter=team_filter,
            samples=samples,
            triggered_by=user_email,
        ))
    except Exception as e:
        print(f"[EVAL] Failed to save run to DB: {e}")

    background_tasks.add_task(_run_eval_background, run_id, eval_type, team_filter, samples)

    await _log_audit("eval_started", _user, target_type="eval", target_id=run_id,
                     details={"eval_type": eval_type, "team": team_filter or "all", "samples": samples})

    return {"status": "accepted", "run_id": run_id, "eval_type": eval_type, "team": team_filter or "all"}


@app.get("/eval/jobs/{run_id}")
def eval_job_status(run_id: str, _user: dict = Depends(_require_role("ADMIN"))):
    """Return current status of a live evaluation run."""
    with _eval_lock:
        job = _eval_jobs.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return job


@app.get("/eval/jobs")
def eval_jobs_list(_user: dict = Depends(_require_role("ADMIN"))):
    """Return all live evaluation jobs."""
    with _eval_lock:
        return list(_eval_jobs.values())


@app.get("/eval/history")
async def eval_history(limit: int = 50, _user: dict = Depends(_require_role("ADMIN"))):
    """Return evaluation run history from MongoDB."""
    try:
        runs = await db_get_eval_runs(limit=limit)
        return runs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval/history/{run_id}")
async def eval_history_detail(run_id: str, _user: dict = Depends(_require_role("ADMIN"))):
    """Return a single evaluation run with full results."""
    run = await db_get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run


# ─────────────────────────────────────────────
# DASHBOARD STATS ENDPOINT
# ─────────────────────────────────────────────

@app.get("/dashboard/stats")
async def dashboard_stats(scope: Optional[str] = None, team_filter: Optional[str] = None, window: Optional[int] = None, _user: dict = Depends(_verify_token)):
    """Return aggregated dashboard statistics from MongoDB, scoped by role.

    Query params:
      scope       — 'me' forces personal view regardless of role
      team_filter — 'DSN' | 'Appli' | 'Outils' narrows results to that RAG team
      window      — number of days for time-windowed stats (default 14)
    """
    from motor.motor_asyncio import AsyncIOMotorClient
    from datetime import timedelta
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGODB_DB", "soprahr_rag")]

    VALID_TEAMS_FILTER = {"DSN", "Appli", "Outils"}

    try:
        history_col = db["analysis_history"]
        users_col = db["users"]
        eval_col = db["eval_runs"]

        role = _user.get("role", "CONSULTANT")
        email = _user.get("sub", "")
        scope_mode = (scope or "").strip().lower()
        days = max(1, min(int(window or 14), 90))

        # ── Validate optional team filter
        tf = (team_filter or "").strip()
        team_extra: dict = {}
        if tf and tf in VALID_TEAMS_FILTER:
            team_extra = {"filters_applied.team": tf}

        # ── Build scope filter based on role (or explicit scope=me)
        scope_filter: dict = {}
        user_doc = await users_col.find_one({"email": email})
        user_id = str(user_doc["_id"]) if user_doc else str(uuid4())

        if scope_mode == "me":
            scope_filter = {"user_id": user_id}
        elif role == "CONSULTANT":
            scope_filter = {"user_id": user_id}
        elif role == "TEAM_LEAD":
            users_json = _load_users()
            user_team = users_json.get(email, {}).get("team", "DSN")
            team_user_ids = []
            async for u in users_col.find({"team": user_team}):
                team_user_ids.append(str(u["_id"]))
            if team_user_ids:
                scope_filter = {"user_id": {"$in": team_user_ids}}
            else:
                scope_filter = {"user_id": "__none__"}
        # ADMIN and MANAGER see everything (scope_filter stays {})

        # Merged filter used for all aggregations
        base_filter = {**scope_filter, **team_extra}

        # ── Total analyses count
        total_analyses = await history_col.count_documents(base_filter)

        # ── Active users (distinct user_ids)
        active_docs = await history_col.aggregate([
            {"$match": base_filter}, {"$group": {"_id": "$user_id"}}
        ]).to_list(None)
        active_users = len(active_docs)

        # ── Total registered users
        if role in ("ADMIN", "MANAGER"):
            total_users = await users_col.count_documents({})
        elif role == "TEAM_LEAD":
            users_json = _load_users()
            user_team = users_json.get(email, {}).get("team", "DSN")
            total_users = await users_col.count_documents({"team": user_team})
        else:
            total_users = 1

        # ── Average response time (ms)
        avg_match = {**base_filter, "execution_time_ms": {"$ne": None}}
        avg_time_pipeline = await history_col.aggregate([
            {"$match": avg_match},
            {"$group": {"_id": None, "avg": {"$avg": "$execution_time_ms"}}},
        ]).to_list(1)
        avg_response_ms = round(avg_time_pipeline[0]["avg"]) if avg_time_pipeline else 0

        # ── User ratings summary
        helpful_count = await history_col.count_documents({**base_filter, "user_rating": {"$gte": 3}})
        not_helpful_count = await history_col.count_documents({**base_filter, "user_rating": {"$lt": 3, "$ne": None}})
        unrated_count = await history_col.count_documents({**base_filter, "$or": [{"user_rating": None}, {"user_rating": {"$exists": False}}]})

        # ── Daily analysis volume — fills gaps with 0 and returns per-team breakdown
        cutoff = datetime.utcnow() - timedelta(days=days - 1)
        cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_match = {**base_filter, "created_at": {"$gte": cutoff}}

        daily_pipeline_raw = await history_col.aggregate([
            {"$match": daily_match},
            {"$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                    "team": "$filters_applied.team",
                },
                "count": {"$sum": 1},
            }},
            {"$sort": {"_id.date": 1}},
        ]).to_list(days * 10)

        today = datetime.utcnow().date()
        date_range = [str(today - timedelta(days=days - 1 - i)) for i in range(days)]

        # Aggregate into per-team and combined daily_volume
        team_counts: dict = {t: {d: 0 for d in date_range} for t in ["DSN", "Appli", "Outils"]}
        combined_counts: dict = {d: 0 for d in date_range}
        for row in daily_pipeline_raw:
            d = row["_id"]["date"]
            t = row["_id"].get("team") or "Unknown"
            c = row["count"]
            if d in combined_counts:
                combined_counts[d] += c
            if t in team_counts and d in team_counts[t]:
                team_counts[t][d] += c

        daily_volume = [{"date": d, "count": combined_counts[d]} for d in date_range]
        daily_volume_by_team = {
            t: [{"date": d, "count": team_counts[t][d]} for d in date_range]
            for t in ["DSN", "Appli", "Outils"]
        }

        # ── Per-team summary (analyses count + avg response per team)
        team_summary_raw = await history_col.aggregate([
            {"$match": scope_filter},
            {"$group": {
                "_id": "$filters_applied.team",
                "count": {"$sum": 1},
                "avg_ms": {"$avg": "$execution_time_ms"},
                "helpful": {"$sum": {"$cond": [{"$gte": ["$user_rating", 3]}, 1, 0]}},
                "not_helpful": {"$sum": {"$cond": [{"$and": [{"$lt": ["$user_rating", 3]}, {"$ne": ["$user_rating", None]}]}, 1, 0]}},
            }},
        ]).to_list(10)
        team_summary = {
            (row["_id"] or "Unknown"): {
                "count": row["count"],
                "avg_ms": round(row["avg_ms"]) if row["avg_ms"] else 0,
                "helpful": row["helpful"],
                "not_helpful": row["not_helpful"],
            }
            for row in team_summary_raw
        }

        # ── Latest evaluation metrics (admin/manager only)
        latest_eval = None
        if role in ("ADMIN", "MANAGER"):
            eval_cursor = eval_col.find({"status": "completed", "metrics": {"$ne": None}}).sort("started_at", -1).limit(1)
            eval_docs = await eval_cursor.to_list(1)
            if eval_docs:
                ev = eval_docs[0]
                latest_eval = {
                    "effective_rate": ev.get("metrics", {}).get("effective_rate", 0),
                    "ref_hit_rate": ev.get("metrics", {}).get("ref_hit_rate", 0),
                    "mean_overlap": ev.get("metrics", {}).get("mean_overlap", 0),
                    "total_questions": ev.get("metrics", {}).get("total_questions", 0),
                    "evaluated_at": ev.get("started_at"),
                }

        # ── Top queries (most common in window)
        top_queries_raw = await history_col.aggregate([
            {"$match": daily_match},
            {"$group": {"_id": "$query", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5},
        ]).to_list(5)
        top_queries = [{"query": q["_id"], "count": q["count"]} for q in top_queries_raw]

        # ── Top failed queries (from audit_logs, window-scoped)
        failed_since = datetime.utcnow() - timedelta(days=days)
        failed_match: dict = {"action": "query_failed", "created_at": {"$gte": failed_since}}
        if role == "CONSULTANT":
            failed_match["actor_email"] = email
        elif role == "TEAM_LEAD":
            users_json = _load_users()
            user_team = users_json.get(email, {}).get("team", "DSN")
            team_emails = [e for e, u in users_json.items() if u.get("team") == user_team]
            if team_emails:
                failed_match["actor_email"] = {"$in": team_emails}

        failed_raw = await db["audit_logs"].aggregate([
            {"$match": failed_match},
            {"$group": {"_id": "$details.query", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5},
        ]).to_list(5)
        top_failed_queries = [{"query": q["_id"] or "(unknown)", "count": q["count"]} for q in failed_raw]

        # ── Consultant-focused payload
        total_rated = helpful_count + not_helpful_count
        helpful_rate = round((helpful_count / total_rated) * 100) if total_rated > 0 else 0

        recent_docs = await history_col.find(base_filter).sort("created_at", -1).limit(5).to_list(5)
        recent_analyses = [
            {
                "id": str(d.get("_id")),
                "query": d.get("query", ""),
                "created_at": d.get("created_at"),
                "user_rating": d.get("user_rating"),
                "execution_time_ms": d.get("execution_time_ms"),
                "total_results": d.get("total_results", 0),
                "team": (d.get("filters_applied") or {}).get("team"),
            }
            for d in recent_docs
        ]

        failed_queries_7d = await db["audit_logs"].count_documents({
            "action": "query_failed",
            "actor_email": email,
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=7)},
        })

        return {
            "total_analyses": total_analyses,
            "active_users": active_users,
            "total_users": total_users,
            "avg_response_ms": avg_response_ms,
            "helpful_count": helpful_count,
            "not_helpful_count": not_helpful_count,
            "unrated_count": unrated_count,
            "daily_volume": daily_volume,
            "daily_volume_by_team": daily_volume_by_team,
            "team_summary": team_summary,
            "latest_eval": latest_eval,
            "top_queries": top_queries,
            "top_failed_queries": top_failed_queries,
            "window_days": days,
            "scope": role,
            "scope_mode": scope_mode or "role",
            "me": {
                "pending_to_rate": unrated_count,
                "helpful_rate": helpful_rate,
                "failed_queries_7d": failed_queries_7d,
                "recent_analyses": recent_analyses,
            },
        }
    finally:
        client.close()


# ─────────────────────────────────────────────
# ANALYSIS RATING ENDPOINT
# ─────────────────────────────────────────────

class RateRequest(BaseModel):
    query: str
    rating: int  # 1-5, where 4-5 = helpful, 1-2 = not helpful

@app.post("/analysis/rate")
async def rate_analysis(body: RateRequest, _user: dict = Depends(_verify_token)):
    """Find the most recent analysis matching the query and update its rating."""
    if body.rating < 1 or body.rating > 5:
        raise HTTPException(status_code=400, detail="rating must be between 1 and 5")

    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGODB_DB", "soprahr_rag")]

    try:
        user_email = _user.get("sub", "")
        user_doc = await db["users"].find_one({"email": user_email})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")

        user_id = str(user_doc["_id"])

        # Find the most recent analysis for this user matching the query
        entry = await db["analysis_history"].find_one(
            {"user_id": user_id, "query": body.query},
            sort=[("created_at", -1)],
        )
        if not entry:
            raise HTTPException(status_code=404, detail="Analysis entry not found")

        await db["analysis_history"].update_one(
            {"_id": entry["_id"]},
            {"$set": {"user_rating": body.rating, "updated_at": datetime.utcnow()}},
        )

        await _log_audit("analysis_rated", _user, target_type="analysis",
                         target_id=str(entry["_id"]),
                         details={"query": body.query[:200], "rating": body.rating})

        # ── Daily threshold check: push admin notification if >5 utile or >5 inutile today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        utile_count = await db["analysis_history"].count_documents({
            "user_rating": {"$gte": 4}, "created_at": {"$gte": today_start}
        })
        inutile_count = await db["analysis_history"].count_documents({
            "user_rating": {"$lte": 2, "$ne": None}, "created_at": {"$gte": today_start}
        })
        notif_col = db["notifications"]
        if utile_count > 5:
            # Only push once per threshold crossing per day
            already = await notif_col.find_one({
                "category": "analysis",
                "title": "Seuil utile depassé",
                "created_at": {"$gte": today_start},
            })
            if not already:
                await create_notifications_for_admins(
                    db, "info", "analysis",
                    "Seuil utile depassé",
                    f"{utile_count} analyses jugées utiles aujourd'hui (> 5).",
                    "audit",
                )
        if inutile_count > 5:
            already = await notif_col.find_one({
                "category": "analysis",
                "title": "Seuil inutile depassé",
                "created_at": {"$gte": today_start},
            })
            if not already:
                await create_notifications_for_admins(
                    db, "alert", "analysis",
                    "Seuil inutile depassé",
                    f"{inutile_count} analyses jugées inutiles aujourd'hui (> 5).",
                    "audit",
                )

        # ── Low-quality rating: notify team lead (same team) + all managers
        if body.rating <= 2:
            consultant_name = user_doc.get("username") or user_email
            consultant_team = user_doc.get("team", "inconnue")
            query_preview = body.query[:100]

            # Single low-quality alert to team lead of that team + all managers
            await create_notifications_for_role_team(
                db, ["TEAM_LEAD"], consultant_team,
                "alert", "analysis",
                "Analyse jugée de faible qualité",
                f'{consultant_name} ({consultant_team}) a noté une analyse {body.rating}/5 : "{query_preview}".',
                "history",
            )
            await create_notifications_for_role_team(
                db, ["MANAGER"], None,
                "alert", "analysis",
                f"Analyse de faible qualité ({consultant_team})",
                f'{consultant_name} ({consultant_team}) a noté une analyse {body.rating}/5 : "{query_preview}".',
                "history",
            )

            # ── Repeated low-quality: check last 7 days for this consultant
            seven_days_ago = datetime.utcnow() - __import__("datetime").timedelta(days=7)
            low_count = await db["analysis_history"].count_documents({
                "user_id": user_id,
                "user_rating": {"$lte": 2},
                "created_at": {"$gte": seven_days_ago},
            })
            if low_count >= 3:
                # Deduplicate: only one coaching alert per consultant per day
                coaching_key_title = f"Coaching requis – {consultant_name}"
                already_coaching = await notif_col.find_one({
                    "title": coaching_key_title,
                    "created_at": {"$gte": today_start},
                })
                if not already_coaching:
                    await create_notifications_for_role_team(
                        db, ["TEAM_LEAD"], consultant_team,
                        "alert", "analysis",
                        coaching_key_title,
                        f"{consultant_name} a obtenu {low_count} analyses jugées de faible qualité ces 7 derniers jours. Un accompagnement est recommandé.",
                        "history",
                    )
                    await create_notifications_for_role_team(
                        db, ["MANAGER"], None,
                        "alert", "analysis",
                        coaching_key_title,
                        f"{consultant_name} ({consultant_team}) : {low_count} analyses de faible qualité en 7 jours. Accompagnement recommandé.",
                        "history",
                    )

        return {"status": "ok", "analysis_id": str(entry["_id"]), "rating": body.rating}
    finally:
        client.close()


# ─────────────────────────────────────────────
# NOTIFICATIONS ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/notifications")
async def get_user_notifications(
    limit: int = 50,
    unread_only: bool = False,
    user: dict = Depends(_verify_token),
):
    """Return notifications for the current user."""
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
    try:
        user_email = user.get("sub", "")
        user_doc = await db["users"].find_one({"email": user_email})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        user_id = str(user_doc["_id"])
        return await get_notifications(user_id, limit=limit, unread_only=unread_only)
    finally:
        client.close()


@app.get("/notifications/unread-count")
async def get_unread_count(user: dict = Depends(_verify_token)):
    """Return the count of unread notifications for the current user."""
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
    try:
        user_email = user.get("sub", "")
        user_doc = await db["users"].find_one({"email": user_email})
        if not user_doc:
            return {"count": 0}
        user_id = str(user_doc["_id"])
        count = await count_unread_notifications(user_id)
        return {"count": count}
    finally:
        client.close()


@app.post("/notifications/{notif_id}/read")
async def mark_notif_read(notif_id: str, user: dict = Depends(_verify_token)):
    """Mark a single notification as read."""
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
    try:
        user_email = user.get("sub", "")
        user_doc = await db["users"].find_one({"email": user_email})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        user_id = str(user_doc["_id"])
        ok = await mark_notification_read(notif_id, user_id)
        return {"status": "ok" if ok else "not_found"}
    finally:
        client.close()


@app.post("/notifications/read-all")
async def mark_all_notifs_read(user: dict = Depends(_verify_token)):
    """Mark all notifications as read for the current user."""
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGODB_DB", "soprahr_rag")]
    try:
        user_email = user.get("sub", "")
        user_doc = await db["users"].find_one({"email": user_email})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        user_id = str(user_doc["_id"])
        count = await mark_all_notifications_read(user_id)
        return {"status": "ok", "updated": count}
    finally:
        client.close()
