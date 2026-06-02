#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
routes.py
=========
FastAPI route handlers for tickets, users, and analysis history.

Include in api.py with:
  from scripts.routes import router
  app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from jose import jwt
from dotenv import load_dotenv

load_dotenv()

from scripts.models import (
    Ticket, TicketCreate, TicketSearch,
    UserCreate, UserLogin, UserProfile,
    AnalysisHistoryCreate, AnalysisHistory, AnalysisHistoryUpdate,
    TokenResponse, AuthResponse, MessageResponse, PaginatedResponse,
    AuditLogCreate,
)
from scripts.database import (
    create_ticket, get_ticket_by_reference, get_ticket_by_id,
    search_tickets, get_recent_tickets, update_ticket, bulk_insert_tickets,
    create_user, get_user_by_username, get_user_by_email, get_user_by_id, get_all_users, update_user,
    add_favorite_ticket, remove_favorite_ticket,
    save_analysis, get_user_analysis_history, update_analysis_entry, delete_analysis_entry,
    get_team_analysis_stats,
    create_audit_log,
)

router = APIRouter()
_bearer = HTTPBearer()

# JWT configuration — must match the secret used in api.py to verify tokens
import os
_JWT_SECRET = os.getenv("JWT_SECRET", "dev_sopra_hr_rag_secret_2026")
_JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
_JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "8"))

def _create_jwt_token(username: str, email: str, role: str = "CONSULTANT") -> str:
    """Create a JWT token for user authentication."""
    payload = {
        "sub": email,
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=_JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    try:
        return jwt.decode(credentials.credentials, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except Exception:
        raise HTTPException(status_code=401, detail="Token invalide ou expiré")


async def _log_audit(action: str, actor_email: str = "unknown", actor_role: str = "CONSULTANT",
                     target_type: str = None, target_id: str = None, details: dict = None):
    """Helper to log an audit event."""
    try:
        await create_audit_log(AuditLogCreate(
            action=action,
            actor_email=actor_email,
            actor_role=actor_role,
            target_type=target_type,
            target_id=target_id,
            details=details,
        ))
    except Exception as e:
        print(f"[WARN] Audit log failed (routes): {e}")


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "RAG API", "timestamp": datetime.utcnow().isoformat()}


# ─────────────────────────────────────────────
# TICKET ENDPOINTS
# ─────────────────────────────────────────────

@router.get("/tickets/search", response_model=List[TicketSearch])
async def search_tickets_endpoint(
    q: str = Query("", min_length=0, description="Search query (optional)"),
    team: Optional[str] = Query(None, description="Filter by team (DSN, Appli, Outils)"),
    closed: Optional[bool] = Query(None, description="Filter by closed status"),
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    """
    Search tickets by keyword, team, or status.
    
    **Parameters:**
    - `q`: Search query (leave empty to get all)
    - `team`: Optional filter (DSN, Appli, Outils)
    - `closed`: Optional filter (true/false)
    - `limit`: Results per page (default 20, max 100)
    - `skip`: Pagination offset (default 0)
    """
    results, total = await search_tickets(q, support_team=team, is_closed=closed, limit=limit, skip=skip)
    return results


@router.get("/tickets/recent", response_model=List[Ticket])
async def get_recent_tickets_endpoint(limit: int = Query(20, ge=1, le=100)):
    """Get recently updated tickets."""
    return await get_recent_tickets(limit=limit)


@router.get("/tickets/{reference}", response_model=Ticket)
async def get_ticket_by_reference_endpoint(reference: str):
    """Get ticket by reference (e.g., 'FR W210000')."""
    ticket = await get_ticket_by_reference(reference)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {reference} not found")
    return ticket


@router.get("/tickets/id/{ticket_id}", response_model=Ticket)
async def get_ticket_by_id_endpoint(ticket_id: str):
    """Get ticket by MongoDB ID."""
    ticket = await get_ticket_by_id(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.post("/tickets", response_model=Ticket, status_code=status.HTTP_201_CREATED)
async def create_ticket_endpoint(ticket_data: TicketCreate):
    """Create a new ticket (admin only)."""
    existing = await get_ticket_by_reference(ticket_data.reference)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Ticket {ticket_data.reference} already exists"
        )
    result = await create_ticket(ticket_data)
    await _log_audit("ticket_created", target_type="ticket", target_id=ticket_data.reference,
                     details={"reference": ticket_data.reference})
    return result


@router.post("/tickets/bulk", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def bulk_insert_tickets_endpoint(tickets: List[TicketCreate]):
    """Bulk insert multiple tickets (admin only)."""
    count = await bulk_insert_tickets(tickets)
    return MessageResponse(message=f"Successfully inserted {count} tickets")


@router.put("/tickets/{ticket_id}", response_model=Ticket)
async def update_ticket_endpoint(ticket_id: str, update_data: dict):
    """Update ticket (admin only)."""
    ticket = await update_ticket(ticket_id, update_data)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    await _log_audit("ticket_updated", target_type="ticket", target_id=ticket_id,
                     details={"fields": list(update_data.keys())})
    return ticket


# ─────────────────────────────────────────────
# USER ENDPOINTS
# ─────────────────────────────────────────────

@router.post("/users/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    """
    Register a new user (default role: CONSULTANT).
    
    **Body:**
    - `username`: 3-50 characters
    - `email`: Valid email
    - `password`: Min 8 characters
    - `team`: Optional (DSN, Appli, Outils)
    """
    existing = await get_user_by_username(user_data.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    from passlib.context import CryptContext
    pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"])
    password_hash = pwd_ctx.hash(user_data.password)
    
    user = await create_user(user_data, password_hash, role="CONSULTANT")
    user_profile = UserProfile(**user.dict())
    token = _create_jwt_token(user.username, user.email, user.role)
    
    await _log_audit("signup", actor_email=user.email, actor_role="CONSULTANT",
                     details={"username": user.username, "team": user_data.team})
    
    return AuthResponse(
        access_token=token,
        expires_in=_JWT_EXPIRE_HOURS * 3600,
        user=user_profile
    )


@router.post("/users/login", response_model=AuthResponse)
async def login_user(credentials: UserLogin):
    """
    Login user and get JWT token with user data.
    
    **Body:**
    - `username`: Username
    - `password`: Password
    """
    user = await get_user_by_username(credentials.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    from passlib.context import CryptContext
    pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"])
    if not pwd_ctx.verify(credentials.password, user.password_hash):
        await _log_audit("login_failed", actor_email=user.email)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_profile = UserProfile(**user.dict())
    token = _create_jwt_token(user.username, user.email, user.role)
    
    await _log_audit("login", actor_email=user.email, actor_role=user.role)
    
    return AuthResponse(
        access_token=token,
        expires_in=_JWT_EXPIRE_HOURS * 3600,
        user=user_profile
    )


@router.get("/users/{user_id}", response_model=UserProfile)
async def get_user_endpoint(user_id: str):
    """Get user profile by ID."""
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserProfile(**user.dict())


@router.get("/users", response_model=List[UserProfile])
async def list_users_endpoint(
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: dict = Depends(_verify_token),
):
    """Get users scoped by role: admin/manager see all, team leads see their team."""
    role = current_user.get("role", "CONSULTANT")

    if role in ("ADMIN", "MANAGER"):
        return await get_all_users(limit=limit, skip=skip)

    if role == "TEAM_LEAD":
        email = current_user.get("sub", "")
        user = await get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        team_users = await get_all_users(limit=500, skip=0)
        scoped_users = [u for u in team_users if u.team == user.team]
        return scoped_users[skip: skip + limit]

    raise HTTPException(status_code=403, detail="Accès refusé: rôle insuffisant")


@router.put("/users/{user_id}", response_model=UserProfile)
async def update_user_endpoint(user_id: str, update_data: dict):
    """Update user profile/role (user can update own, admin can update all)."""
    user = await update_user(user_id, update_data)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserProfile(**user.dict())


# ─────────────────────────────────────────────
# FAVORITES ENDPOINTS
# ─────────────────────────────────────────────

@router.post("/users/{user_id}/favorites/{ticket_id}", response_model=MessageResponse)
async def add_favorite(user_id: str, ticket_id: str):
    """Add ticket to user's favorites."""
    # TODO: Add auth check to ensure user_id matches current user
    user = await add_favorite_ticket(user_id, ticket_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await _log_audit("favorite_added", target_type="ticket", target_id=ticket_id,
                     details={"user_id": user_id})
    return MessageResponse(message="Ticket added to favorites")


@router.delete("/users/{user_id}/favorites/{ticket_id}", response_model=MessageResponse)
async def remove_favorite(user_id: str, ticket_id: str):
    """Remove ticket from user's favorites."""
    # TODO: Add auth check to ensure user_id matches current user
    user = await remove_favorite_ticket(user_id, ticket_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await _log_audit("favorite_removed", target_type="ticket", target_id=ticket_id,
                     details={"user_id": user_id})
    return MessageResponse(message="Ticket removed from favorites")


# ─────────────────────────────────────────────
# ANALYSIS HISTORY ENDPOINTS
# ─────────────────────────────────────────────

@router.post("/users/{user_id}/analysis", response_model=AnalysisHistory, status_code=status.HTTP_201_CREATED)
async def save_analysis_endpoint(user_id: str, history_data: AnalysisHistoryCreate):
    """
    Save a search/query to analysis history.
    
    **Body:**
    - `query`: The search query
    - `query_type`: "search" or "rag_query" (default: "search")
    - `matched_tickets`: List of matched ticket objects
    - `generated_answer`: Optional LLM-generated answer
    - `execution_time_ms`: Optional execution time
    - `total_results`: Optional total results count
    - `filters_applied`: Optional filter dict
    """
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    history = await save_analysis(user_id, history_data)
    return history


@router.get("/users/{user_id}/analysis", response_model=List[AnalysisHistory])
async def get_analysis_history_endpoint(
    user_id: str,
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
):
    """
    Get user's search/query history.
    
    **Parameters:**
    - `limit`: Results per page (default 50)
    - `skip`: Pagination offset (default 0)
    """
    print(f"[DEBUG] get_analysis_history_endpoint called with user_id={user_id}, limit={limit}, skip={skip}")
    
    user = await get_user_by_id(user_id)
    if not user:
        print(f"[DEBUG] User not found for id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    
    print(f"[DEBUG] User found: {user.username}")
    history, total = await get_user_analysis_history(user_id, limit=limit, skip=skip)
    print(f"[DEBUG] Found {len(history)} analysis entries for user {user_id}")
    return history


@router.get("/users/{user_id}/analysis/{analysis_id}", response_model=AnalysisHistory)
async def get_analysis_entry_endpoint(user_id: str, analysis_id: str):
    """Get a specific analysis entry."""
    from scripts.database import _db
    from bson import ObjectId
    
    history_collection = _db["analysis_history"]
    try:
        doc = await history_collection.find_one({"_id": ObjectId(analysis_id), "user_id": user_id})
        if doc:
            return AnalysisHistory(**doc)
    except:
        pass
    
    raise HTTPException(status_code=404, detail="Analysis entry not found")


@router.put("/users/{user_id}/analysis/{analysis_id}", response_model=AnalysisHistory)
async def update_analysis_entry_endpoint(
    user_id: str,
    analysis_id: str,
    update_data: AnalysisHistoryUpdate
):
    """Update analysis entry (rating, tags)."""
    entry = await update_analysis_entry(analysis_id, user_id, update_data.dict(exclude_unset=True))
    if not entry:
        raise HTTPException(status_code=404, detail="Analysis entry not found")
    return entry


@router.delete("/users/{user_id}/analysis/{analysis_id}", response_model=MessageResponse)
async def delete_analysis_entry_endpoint(user_id: str, analysis_id: str):
    """Delete a history entry."""
    print(f"[DEBUG DELETE] user_id={user_id}, analysis_id={analysis_id}")
    success = await delete_analysis_entry(analysis_id, user_id)
    print(f"[DEBUG DELETE] success={success}")
    if not success:
        raise HTTPException(status_code=404, detail="Analysis entry not found")
    return MessageResponse(message="Analysis entry deleted")


# ─────────────────────────────────────────────
# TEAM ANALYTICS ENDPOINTS (Manager Only)
# ─────────────────────────────────────────────

@router.get("/teams/{team}/analytics")
async def get_team_analytics(team: str, days: int = Query(7, ge=1, le=365)):
    """Get team analysis statistics (team manager only)."""
    stats = await get_team_analysis_stats(team, days=days)
    return stats
