"""Tests for ask-before-use tool approval (GUI permission prompts)."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec


class _Tool(BaseTool):
    def __init__(self, name: str, *, requires_confirmation: bool = False) -> None:
        self._name = name
        self._confirm = requires_confirmation
        self.calls: list[dict] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=f"{self._name} tool",
            requires_confirmation=self._confirm,
        )

    def execute(self, **params: Any) -> ToolResult:
        self.calls.append(params)
        return ToolResult(tool_name=self._name, content="ran", success=True)


def _call(name: str, **args: Any) -> ToolCall:
    return ToolCall(id="1", name=name, arguments=json.dumps(args))


# --- ToolExecutor -----------------------------------------------------------


class TestExecutorApproval:
    def test_listed_tool_asks_and_runs_when_approved(self):
        tool = _Tool("file_read")
        executor = ToolExecutor([tool])
        asked: list = []
        executor.enable_approval(
            lambda name, args: asked.append((name, args)) or True, ["file_read"]
        )
        result = executor.execute(_call("file_read", path="~/notes.md"))
        assert result.success is True
        assert asked == [("file_read", {"path": "~/notes.md"})]
        assert tool.calls == [{"path": "~/notes.md"}]

    def test_listed_tool_denied_does_not_run(self):
        tool = _Tool("file_read")
        executor = ToolExecutor([tool])
        executor.enable_approval(lambda name, args: False, ["file_read"])
        result = executor.execute(_call("file_read", path="/etc/hosts"))
        assert result.success is False
        assert "denied" in result.content.lower()
        assert tool.calls == []

    def test_unlisted_tool_runs_without_asking(self):
        tool = _Tool("think")
        executor = ToolExecutor([tool])
        executor.enable_approval(lambda *a: pytest.fail("asked"), ["file_read"])
        assert executor.execute(_call("think", thought="x")).success is True

    def test_wildcard_asks_for_every_tool(self):
        tool = _Tool("web_search")
        executor = ToolExecutor([tool])
        asked: list = []
        executor.enable_approval(lambda n, a: asked.append(n) or True, ["*"])
        executor.execute(_call("web_search", query="weather"))
        assert asked == ["web_search"]

    def test_confirmation_tools_use_approval_instead_of_failing(self):
        """shell_exec-style tools fail closed without a callback; with approval
        enabled they ask instead, even if not listed."""
        tool = _Tool("shell_exec", requires_confirmation=True)
        executor = ToolExecutor([tool])
        assert executor.execute(_call("shell_exec", command="ls")).success is False
        executor.enable_approval(lambda n, a: True, [])
        assert executor.execute(_call("shell_exec", command="ls")).success is True

    def test_approval_callback_error_denies(self):
        tool = _Tool("file_read")
        executor = ToolExecutor([tool])

        def boom(name, args):
            raise RuntimeError("broker gone")

        executor.enable_approval(boom, ["file_read"])
        assert executor.execute(_call("file_read", path="x")).success is False
        assert tool.calls == []


# --- ApprovalBroker ---------------------------------------------------------


class TestApprovalBroker:
    def _broker(self, timeout=5.0):
        from openjarvis.server.tool_approvals import ToolApprovalBroker

        return ToolApprovalBroker(timeout=timeout)

    def _resolve_when_pending(self, broker, approved):
        def worker():
            for _ in range(200):
                pending = broker.pending()
                if pending:
                    broker.resolve(pending[0]["id"], approved)
                    return
                time.sleep(0.01)

        thread = threading.Thread(target=worker)
        thread.start()
        return thread

    def test_request_blocks_until_approved(self):
        broker = self._broker()
        thread = self._resolve_when_pending(broker, True)
        assert broker.request("file_read", {"path": "~/a.md"}) is True
        thread.join()
        assert broker.pending() == []

    def test_request_denied(self):
        broker = self._broker()
        thread = self._resolve_when_pending(broker, False)
        assert broker.request("shell_exec", {"command": "rm -rf /tmp/x"}) is False
        thread.join()

    def test_request_times_out_as_denied(self):
        broker = self._broker(timeout=0.05)
        assert broker.request("file_read", {"path": "x"}) is False
        assert broker.pending() == []

    def test_pending_describes_action_readably(self):
        broker = self._broker(timeout=0.3)
        seen: list = []

        def peek():
            for _ in range(100):
                if broker.pending():
                    seen.extend(broker.pending())
                    broker.resolve(seen[0]["id"], False)
                    return
                time.sleep(0.005)

        thread = threading.Thread(target=peek)
        thread.start()
        broker.request("file_read", {"path": "~/Documents/lease.pdf"})
        thread.join()
        item = seen[0]
        assert item["tool"] == "file_read"
        assert item["summary"] == "Read file ~/Documents/lease.pdf"
        assert item["arguments"] == {"path": "~/Documents/lease.pdf"}

    @pytest.mark.parametrize(
        "tool, args, summary",
        [
            ("shell_exec", {"command": "ls ~"}, "Run command: ls ~"),
            ("file_write", {"path": "/tmp/n.md"}, "Write file /tmp/n.md"),
            ("web_search", {"query": "weather"}, "Search the web for: weather"),
            ("custom", {"a": 1}, "Use tool 'custom'"),
        ],
    )
    def test_summaries(self, tool, args, summary):
        from openjarvis.server.tool_approvals import summarize_tool_call

        assert summarize_tool_call(tool, args) == summary

    def test_resolve_unknown_returns_false(self):
        assert self._broker().resolve("nope", True) is False


# --- HTTP routes --------------------------------------------------------------


def test_approval_routes_roundtrip():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from openjarvis.server.tool_approvals import ToolApprovalBroker, router

    app = FastAPI()
    app.include_router(router)
    broker = ToolApprovalBroker(timeout=5.0)
    app.state.tool_approvals = broker
    client = TestClient(app)

    result: list = []
    thread = threading.Thread(
        target=lambda: result.append(broker.request("file_read", {"path": "x"}))
    )
    thread.start()
    for _ in range(200):
        pending = client.get("/v1/tool-approvals/pending").json()["approvals"]
        if pending:
            break
        time.sleep(0.01)
    assert pending[0]["summary"] == "Read file x"
    resp = client.post(
        f"/v1/tool-approvals/{pending[0]['id']}", json={"approved": True}
    )
    assert resp.status_code == 200
    thread.join()
    assert result == [True]
    assert (
        client.post("/v1/tool-approvals/missing", json={"approved": True}).status_code
        == 404
    )


def test_approval_routes_without_broker():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from openjarvis.server.tool_approvals import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.get("/v1/tool-approvals/pending").json() == {
        "approvals": [],
        "enabled": False,
    }


# --- Wiring -----------------------------------------------------------------


def test_enable_tool_approval_wires_real_agent():
    from unittest.mock import MagicMock

    from openjarvis.agents.orchestrator import OrchestratorAgent
    from openjarvis.server.tool_approvals import (
        ToolApprovalBroker,
        enable_tool_approval,
    )

    tool = _Tool("file_read")
    agent = OrchestratorAgent(MagicMock(), "m", tools=[tool])
    broker = ToolApprovalBroker(timeout=0.01)
    assert enable_tool_approval(agent, broker, "file_read, shell_exec") is True
    denied = agent._executor.execute(_call("file_read", path="x"))  # times out
    assert denied.success is False and tool.calls == []


def test_enable_tool_approval_ignores_toolless_agent():
    from openjarvis.server.tool_approvals import (
        ToolApprovalBroker,
        enable_tool_approval,
    )

    assert enable_tool_approval(object(), ToolApprovalBroker(), ["*"]) is False


def test_require_approval_config_parses_list_and_string():
    from openjarvis.server.tool_approvals import parse_tool_list

    assert parse_tool_list(["file_read", " shell_exec "]) == ["file_read", "shell_exec"]
    assert parse_tool_list("file_read, web_search") == ["file_read", "web_search"]
    assert parse_tool_list("") == []


def test_all_routes_expose_tool_approvals():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI

    from openjarvis.server.api_routes import include_all_routes

    app = FastAPI()
    include_all_routes(app)
    paths = {route.path for route in app.routes}
    assert "/v1/tool-approvals/pending" in paths


# --- Terminal approval (jarvis chat / jarvis ask) -----------------------------


def test_terminal_approval_prompts_and_reads_answer(monkeypatch):
    from openjarvis.tools.approval import terminal_approval_callback

    printed: list = []
    console = type("C", (), {"print": lambda self, *a, **k: printed.append(a[0])})()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["y", "n", ""])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    ask = terminal_approval_callback(console)
    assert ask("file_read", {"path": "~/a.md"}) is True
    assert ask("file_read", {"path": "~/a.md"}) is False
    assert ask("shell_exec", {"command": "ls"}) is False  # default is No
    assert "Read file ~/a.md" in printed[0]


def test_terminal_approval_denies_without_a_terminal(monkeypatch):
    from openjarvis.tools.approval import terminal_approval_callback

    console = type("C", (), {"print": lambda self, *a, **k: None})()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("prompted"))
    assert terminal_approval_callback(console)("file_read", {"path": "x"}) is False


def test_apply_approval_from_config():
    from types import SimpleNamespace

    from openjarvis.tools.approval import apply_approval_from_config

    tool = _Tool("file_read")
    executor_holder = SimpleNamespace(_executor=ToolExecutor([tool]))
    off = SimpleNamespace(tools=SimpleNamespace(require_approval=""))
    on = SimpleNamespace(tools=SimpleNamespace(require_approval="file_read"))
    assert apply_approval_from_config(executor_holder, off, lambda n, a: True) is False
    assert apply_approval_from_config(executor_holder, on, lambda n, a: False) is True
    assert (
        executor_holder._executor.execute(_call("file_read", path="x")).success is False
    )
