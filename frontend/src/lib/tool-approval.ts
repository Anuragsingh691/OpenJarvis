import type { ToolApproval } from './api';

/** Argument shown in the one-line summary, so details don't repeat it. */
const SUMMARY_ARG: Record<string, string> = {
  file_read: 'path',
  file_write: 'path',
  apply_patch: 'path',
  shell_exec: 'command',
  web_search: 'query',
  http_request: 'url',
};

/**
 * Extra detail for an approval card: code to be run, or any arguments not
 * already in the summary. Returns null when there's nothing more to show.
 */
export function approvalDetail(approval: Pick<ToolApproval, 'tool' | 'arguments'>): string | null {
  const args = { ...approval.arguments };
  if (approval.tool === 'code_interpreter' && typeof args.code === 'string') {
    return args.code;
  }
  if (approval.tool === 'file_write' && typeof args.content === 'string') {
    const content = args.content;
    return content.length > 600 ? `${content.slice(0, 600)}…` : content;
  }
  const shown = SUMMARY_ARG[approval.tool];
  if (shown) delete args[shown];
  return Object.keys(args).length ? JSON.stringify(args, null, 2) : null;
}

/** Whole seconds until the server auto-denies (never negative). */
export function secondsLeft(expiresAt: number, nowMs: number = Date.now()): number {
  return Math.max(0, Math.ceil(expiresAt - nowMs / 1000));
}
