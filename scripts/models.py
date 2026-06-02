#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models.py
=========
Pydantic data models for MongoDB collections (users, tickets, analysis history).
With role-based access control (Admin, TeamManager, Consultant).
"""

from datetime import datetime
from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field, EmailStr, field_validator


# ─────────────────────────────────────────────
# ENUMS / LITERALS
# ─────────────────────────────────────────────

ROLE = Literal["ADMIN", "MANAGER", "TEAM_LEAD", "CONSULTANT"]
SUPPORT_TEAM = Literal["DSN", "Appli", "Outils"]
QUERY_TYPE = Literal["search", "rag_query"]


# ─────────────────────────────────────────────
# TICKET MODELS
# ─────────────────────────────────────────────

class ConversationEntry(BaseModel):
    """Single message/action in a ticket conversation."""
    timestamp: Union[str, datetime]
    actor: str  # "client", "support", or other values from raw data
    author: Optional[str] = ""
    type: Optional[str] = "message"
    action: Optional[str] = None
    text: Optional[str] = None    # message body (mapped from raw "content" field)
    content: Optional[str] = None  # kept for backwards-compatibility with raw JSON

    @field_validator("timestamp", mode="before")
    @classmethod
    def coerce_timestamp(cls, v):
        if isinstance(v, datetime):
            return v.strftime("%d/%m/%Y %H:%M")
        return v


class TicketCreate(BaseModel):
    """Schema for creating/updating a ticket."""
    reference: str = Field(..., description="e.g., 'FR W210000'")
    title: str
    source_file: Optional[str] = None
    version: Optional[str] = None
    system: Optional[str] = None
    site_env: Optional[str] = None
    support_team: Optional[SUPPORT_TEAM] = None
    team_source: Optional[str] = None
    closing_teamcode: Optional[str] = None
    closing_status_code: Optional[str] = None
    closing_status_explanation: Optional[str] = None
    closing_level: Optional[str] = None
    is_closed: bool = False
    description: str
    resolution: Optional[str] = None
    espdsn_version: Optional[str] = None
    patches: List[str] = Field(default_factory=list)
    conversation: List[ConversationEntry] = Field(default_factory=list)
    message_count: int = 0
    confidence: Optional[float] = None
    content_hash: Optional[str] = None

    @field_validator("support_team", mode="before")
    @classmethod
    def _coerce_unknown_team(cls, v):
        # Tickets in raw data sometimes have support_team='Unknown' or '';
        # treat those as missing rather than failing validation.
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in ("", "unknown", "none", "null"):
            return None
        return v


class Ticket(TicketCreate):
    """Ticket model with MongoDB metadata."""
    id: str = Field(alias="_id")
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True


class TicketSearch(BaseModel):
    """Schema for ticket search results."""
    id: str
    reference: str
    title: str
    support_team: Optional[str]
    is_closed: bool
    description: str
    similarity_score: Optional[float] = None


# ─────────────────────────────────────────────
# USER MODELS
# ─────────────────────────────────────────────

class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    team: Optional[SUPPORT_TEAM] = None


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str
    password: str


class UserProfile(BaseModel):
    """User profile information (public)."""
    id: str = Field(alias="_id")
    username: str
    email: str
    role: ROLE
    team: Optional[SUPPORT_TEAM] = None
    created_at: datetime
    updated_at: datetime
    favorite_tickets: List[str] = Field(default_factory=list)
    is_active: bool = True

    class Config:
        populate_by_name = True


class User(UserProfile):
    """Full user model with sensitive data (internal use only)."""
    password_hash: str


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    team: Optional[SUPPORT_TEAM] = None
    # Admin-only fields
    role: Optional[ROLE] = None
    is_active: Optional[bool] = None


class AdminUserCreate(BaseModel):
    """Schema for admin-created users (no self-registration)."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    team: Optional[SUPPORT_TEAM] = None
    role: ROLE = "CONSULTANT"


# ─────────────────────────────────────────────
# ANALYSIS HISTORY MODELS
# ─────────────────────────────────────────────

class MatchedTicket(BaseModel):
    """Ticket matched in analysis result."""
    ticket_id: str
    reference: str
    title: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    support_team: Optional[str]


class AnalysisHistoryCreate(BaseModel):
    """Schema for saving analysis/search history."""
    query: str
    query_type: QUERY_TYPE = "search"
    matched_tickets: List[MatchedTicket]
    generated_answer: Optional[str] = None
    execution_time_ms: Optional[int] = None
    total_results: Optional[int] = None
    filters_applied: Optional[dict] = None


class AnalysisHistory(AnalysisHistoryCreate):
    """Saved analysis history with metadata."""
    id: str = Field(alias="_id")
    user_id: str
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True


class AnalysisHistoryUpdate(BaseModel):
    """Schema for updating analysis history (e.g., rating)."""
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    tags: Optional[List[str]] = None


# ─────────────────────────────────────────────
# PIPELINE JOB MODELS
# ─────────────────────────────────────────────

PIPELINE_STATUS = Literal["processing", "completed", "failed"]


class PipelineJobCreate(BaseModel):
    """Schema for creating a pipeline job record."""
    job_id: str
    file_name: str
    mode: Literal["raw", "simple"]
    triggered_by: str  # user email


class PipelineJobLog(BaseModel):
    """A timestamped log entry for a pipeline job."""
    timestamp: datetime
    message: str
    level: Literal["info", "warn", "error"] = "info"


class PipelineJob(BaseModel):
    """Full pipeline job record stored in MongoDB."""
    id: str = Field(alias="_id")
    job_id: str
    file_name: str
    mode: Literal["raw", "simple"]
    triggered_by: str
    status: PIPELINE_STATUS = "processing"
    progress: int = 0
    tickets_added: int = 0
    logs: List[PipelineJobLog] = Field(default_factory=list)
    started_at: datetime
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        populate_by_name = True


# ─────────────────────────────────────────────
# EVALUATION MODELS
# ─────────────────────────────────────────────

EVAL_STATUS = Literal["processing", "completed", "failed"]
EVAL_TYPE = Literal["agent", "hybrid", "vectorless"]


class EvalTeamMetrics(BaseModel):
    """Per-team breakdown from an evaluation run."""
    questions: int = 0
    ref_hit_rate: float = 0.0
    effective_rate: float = 0.0
    mean_overlap: float = 0.0


class EvalMetrics(BaseModel):
    """Top-level metrics from an evaluation run."""
    total_questions: int = 0
    ref_hit_rate: float = 0.0
    resolution_rate: float = 0.0
    effective_rate: float = 0.0
    mean_overlap: float = 0.0
    mean_latency_s: float = 0.0
    errors: int = 0


class EvalResultItem(BaseModel):
    """Single question result in an evaluation run."""
    id: str
    team: str = ""
    question: str = ""
    expected_ref: str = ""
    expected_answer: str = ""
    cited_refs: List[str] = Field(default_factory=list)
    ref_hit: bool = False
    has_resolution: bool = False
    overlap: float = 0.0
    effective: bool = False
    latency_s: float = 0.0
    error: Optional[str] = None


class EvalRunCreate(BaseModel):
    """Schema for creating an evaluation run record."""
    run_id: str
    eval_type: EVAL_TYPE = "agent"
    team_filter: Optional[str] = None
    samples: Optional[int] = None
    triggered_by: str  # user email


class EvalRun(BaseModel):
    """Full evaluation run record stored in MongoDB."""
    id: str = Field(alias="_id")
    run_id: str
    eval_type: EVAL_TYPE = "agent"
    team_filter: Optional[str] = None
    samples: Optional[int] = None
    triggered_by: str
    status: EVAL_STATUS = "processing"
    progress: int = 0
    metrics: Optional[dict] = None
    by_team: Optional[dict] = None
    results: List[dict] = Field(default_factory=list)
    logs: List[PipelineJobLog] = Field(default_factory=list)  # reuse log model
    started_at: datetime
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        populate_by_name = True


# ─────────────────────────────────────────────
# API RESPONSE MODELS
# ─────────────────────────────────────────────

class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AuthResponse(BaseModel):
    """Combined auth response with token and user data."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: 'UserProfile'


class MessageResponse(BaseModel):
    """Generic success message."""
    message: str


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    status_code: int


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    data: List[dict]
    total: int
    limit: int
    skip: int
    has_more: bool


# ─────────────────────────────────────────────
# AUDIT LOG MODELS
# ─────────────────────────────────────────────

AUDIT_ACTION = Literal[
    # Auth
    "signup", "login", "login_failed", "logout",
    # User management
    "user_created", "user_updated", "user_deleted", "role_changed",
    # Queries & analysis
    "query_executed", "query_failed", "analysis_rated",
    # Pipeline
    "pipeline_started", "pipeline_completed", "pipeline_failed",
    # Evaluation
    "eval_started", "eval_completed", "eval_failed",
    # Tickets
    "ticket_created", "ticket_updated", "ticket_searched",
    "tickets_reloaded",
    # Favorites
    "favorite_added", "favorite_removed",
    # Settings / admin
    "settings_changed", "admin_users_viewed", "audit_logs_viewed",
]


class AuditLogCreate(BaseModel):
    """Schema for creating an audit log entry."""
    action: str
    actor_email: str
    actor_role: str
    target_type: Optional[str] = None  # "user", "ticket", "pipeline", "eval"
    target_id: Optional[str] = None
    details: Optional[dict] = None
    ip_address: Optional[str] = None


class AuditLog(AuditLogCreate):
    """Full audit log record stored in MongoDB."""
    id: str = Field(alias="_id")
    created_at: datetime

    class Config:
        populate_by_name = True


# ─────────────────────────────────────────────
# NOTIFICATION MODELS
# ─────────────────────────────────────────────

NOTIF_TYPE = Literal["alert", "success", "info"]
NOTIF_CATEGORY = Literal["analysis", "evaluation", "pipeline", "system"]


class NotificationCreate(BaseModel):
    """Schema for creating a notification."""
    user_id: str  # recipient user _id (str); use "BROADCAST_ADMIN" for all admins
    type: NOTIF_TYPE = "info"
    category: NOTIF_CATEGORY = "system"
    title: str
    message: str
    target_tab: Optional[str] = None  # frontend tab to navigate to on click


class AppNotification(BaseModel):
    """Full notification record stored in MongoDB."""
    id: str = Field(alias="_id")
    user_id: str
    type: NOTIF_TYPE = "info"
    category: NOTIF_CATEGORY = "system"
    title: str
    message: str
    target_tab: Optional[str] = None
    read: bool = False
    created_at: datetime

    class Config:
        populate_by_name = True
