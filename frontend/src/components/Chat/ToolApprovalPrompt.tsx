import { useCallback, useEffect, useState } from 'react';
import { ShieldAlert } from 'lucide-react';
import { toast } from 'sonner';
import { decideToolApproval, fetchToolApprovals, type ToolApproval } from '../../lib/api';
import { approvalDetail, secondsLeft } from '../../lib/tool-approval';

const POLL_MS = 700;

/**
 * Inline "Jarvis wants to …  Allow / Deny" cards for tool calls that need the
 * user's permission ([tools] require_approval). Polls only while a reply is
 * being generated, since that is the only time the agent can be waiting.
 */
export function ToolApprovalPrompt({ active }: { active: boolean }) {
  const [approvals, setApprovals] = useState<ToolApproval[]>([]);
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [, setTick] = useState(0);

  useEffect(() => {
    if (!active) {
      setApprovals([]);
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const pending = await fetchToolApprovals();
        if (!cancelled) setApprovals(pending);
      } catch {
        // server restarting; try again on the next tick
      }
    };
    void load();
    const id = setInterval(() => {
      void load();
      setTick((t) => t + 1); // refresh countdowns
    }, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [active]);

  const decide = useCallback(async (approval: ToolApproval, approved: boolean) => {
    setBusy((b) => ({ ...b, [approval.id]: true }));
    try {
      await decideToolApproval(approval.id, approved);
      setApprovals((list) => list.filter((a) => a.id !== approval.id));
    } catch {
      toast.error('That request already expired');
      setApprovals((list) => list.filter((a) => a.id !== approval.id));
    } finally {
      setBusy((b) => ({ ...b, [approval.id]: false }));
    }
  }, []);

  if (!approvals.length) return null;

  return (
    <div className="mb-2 flex flex-col gap-2" role="region" aria-label="Permission requests">
      {approvals.map((approval) => {
        const detail = approvalDetail(approval);
        const disabled = !!busy[approval.id];
        return (
          <div
            key={approval.id}
            className="rounded-xl px-4 py-3 text-sm"
            style={{
              background: 'var(--color-bg-secondary)',
              border: '1px solid var(--color-warning)',
            }}
          >
            <div className="flex items-start gap-2">
              <ShieldAlert size={16} className="mt-0.5 shrink-0" style={{ color: 'var(--color-warning)' }} />
              <div className="flex-1 min-w-0">
                <div style={{ color: 'var(--color-text)' }}>
                  Jarvis wants to: <span className="font-medium break-all">{approval.summary}</span>
                </div>
                {detail && (
                  <pre
                    className="mt-2 max-h-40 overflow-auto rounded-lg p-2 text-xs whitespace-pre-wrap break-all"
                    style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-text-secondary)' }}
                  >
                    {detail}
                  </pre>
                )}
                <div className="mt-2 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => decide(approval, true)}
                    disabled={disabled}
                    className="px-3 py-1 rounded-lg text-xs font-medium cursor-pointer disabled:opacity-50"
                    style={{ background: 'var(--color-accent)', color: 'white' }}
                  >
                    Allow
                  </button>
                  <button
                    type="button"
                    onClick={() => decide(approval, false)}
                    disabled={disabled}
                    className="px-3 py-1 rounded-lg text-xs font-medium cursor-pointer disabled:opacity-50"
                    style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-text)' }}
                  >
                    Deny
                  </button>
                  <span className="ml-auto text-[11px]" style={{ color: 'var(--color-text-tertiary)' }}>
                    Auto-deny in {secondsLeft(approval.expires_at)}s
                  </span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
