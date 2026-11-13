"""Ask-before-use tool approval shared by the CLI and the API server.

``[tools] require_approval`` lists tools ("*" = all) that may only run after
the user agrees. The server asks through the GUI (see
``openjarvis.server.tool_approvals``); ``jarvis chat`` / ``jarvis ask`` ask
in the terminal via :func:`terminal_approval_callback`.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List

ApprovalCallback = Callable[[str, Dict[str, Any]], bool]

_SUMMARIES = {
    "file_read": ("Read file", "path"),
    "file_write": ("Write file", "path"),
    "apply_patch": ("Edit file", "path"),
    "shell_exec": ("Run command:", "command"),
    "web_search": ("Search the web for:", "query"),
    "http_request": ("Fetch URL", "url"),
    "code_interpreter": ("Run Python code", None),
}


def summarize_tool_call(tool: str, arguments: Dict[str, Any]) -> str:
    """One-line, human-readable description of a tool call."""
    verb, key = _SUMMARIES.get(tool, (None, None))
    if verb is None:
        return f"Use tool '{tool}'"
    value = arguments.get(key) if key else None
    return f"{verb} {value}" if value else verb


def parse_tool_list(value: Any) -> List[str]:
    """Accept a TOML list or a comma-separated string of tool names."""
    if not value:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [str(v).strip() for v in value if str(v).strip()]


def apply_approval_from_config(
    agent: Any, config: Any, callback: ApprovalCallback
) -> bool:
    """Enable approval on ``agent`` per ``config``. Returns True if enabled."""
    tools = parse_tool_list(
        getattr(getattr(config, "tools", None), "require_approval", "")
    )
    executor = getattr(agent, "_executor", None)
    enable = getattr(executor, "enable_approval", None)
    if not tools or not callable(enable):
        return False
    enable(callback, tools)
    return True


def terminal_approval_callback(console: Any) -> ApprovalCallback:
    """Ask "Allow …? [y/N]" in the terminal; deny when nobody can answer."""

    def ask(tool: str, arguments: Dict[str, Any]) -> bool:
        summary = summarize_tool_call(tool, arguments)
        if not sys.stdin.isatty():
            console.print(
                "[yellow]Denied (no terminal to ask):[/yellow] "
                f"Jarvis wanted to: {summary}"
            )
            return False
        console.print(
            f"[yellow]Permission:[/yellow] Jarvis wants to: [bold]{summary}[/bold]"
            " — allow? [y/N] ",
            end="",
        )
        try:
            return input().strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    return ask


__all__ = [
    "apply_approval_from_config",
    "parse_tool_list",
    "summarize_tool_call",
    "terminal_approval_callback",
]
