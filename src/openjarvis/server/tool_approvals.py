"""Ask-before-use approval for tools called by the server's chat agent.

When ``[tools] require_approval`` is set, the agent's tool executor calls
:meth:`ToolApprovalBroker.request` before running a listed tool. The call
blocks (agents run in worker threads) until the user answers in the GUI via
the routes below, or until it times out, which counts as a denial.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - fastapi is a server dependency
    raise ImportError("fastapi is required for tool approval routes")

from openjarvis.tools.approval import parse_tool_list, summarize_tool_call

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass
class _Pending:
    id: str
    tool: str
    arguments: Dict[str, Any]
    created_at: float
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False


class ToolApprovalBroker:
    """In-process queue of tool calls awaiting the user's decision."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self._lock = threading.Lock()
        self._pending: Dict[str, _Pending] = {}

    def request(self, tool: str, arguments: Dict[str, Any]) -> bool:
        """Block until the user approves or denies; timeout denies."""
        item = _Pending(
            id=uuid.uuid4().hex,
            tool=tool,
            arguments=dict(arguments),
            created_at=time.time(),
        )
        with self._lock:
            self._pending[item.id] = item
        logger.info("Tool approval requested: %s", tool)
        try:
            answered = item.event.wait(self.timeout)
        finally:
            with self._lock:
                self._pending.pop(item.id, None)
        if not answered:
            logger.info("Tool approval timed out: %s", tool)
            return False
        logger.info("Tool %s: %s", "approved" if item.approved else "denied", tool)
        return item.approved

    def resolve(self, approval_id: str, approved: bool) -> bool:
        """Answer a pending request. Returns False if it no longer exists."""
        with self._lock:
            item = self._pending.get(approval_id)
        if item is None:
            return False
        item.approved = bool(approved)
        item.event.set()
        return True

    def pending(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = sorted(self._pending.values(), key=lambda p: p.created_at)
        return [
            {
                "id": p.id,
                "tool": p.tool,
                "summary": summarize_tool_call(p.tool, p.arguments),
                "arguments": p.arguments,
                "created_at": p.created_at,
                "expires_at": p.created_at + self.timeout,
            }
            for p in items
        ]


def enable_tool_approval(agent: Any, broker: ToolApprovalBroker, tools: Any) -> bool:
    """Route ``agent``'s tool calls through ``broker``. Returns True if wired."""
    executor = getattr(agent, "_executor", None)
    enable = getattr(executor, "enable_approval", None)
    if not callable(enable):
        return False
    enable(broker.request, parse_tool_list(tools))
    return True


router = APIRouter(tags=["tool-approvals"])


class _Decision(BaseModel):
    approved: bool


def _broker(request: Request) -> Optional[ToolApprovalBroker]:
    return getattr(request.app.state, "tool_approvals", None)


@router.get("/v1/tool-approvals/pending")
async def list_tool_approvals(request: Request) -> Dict[str, Any]:
    broker = _broker(request)
    if broker is None:
        return {"approvals": [], "enabled": False}
    return {"approvals": broker.pending(), "enabled": True}


@router.post("/v1/tool-approvals/{approval_id}")
async def decide_tool_approval(
    approval_id: str, decision: _Decision, request: Request
) -> Dict[str, Any]:
    broker = _broker(request)
    if broker is None or not broker.resolve(approval_id, decision.approved):
        raise HTTPException(status_code=404, detail="No such pending approval")
    return {"id": approval_id, "approved": decision.approved}


__all__ = [
    "ToolApprovalBroker",
    "enable_tool_approval",
    "parse_tool_list",
    "router",
    "summarize_tool_call",
]
