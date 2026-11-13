import { describe, expect, it } from 'vitest';
import { approvalDetail, secondsLeft } from './tool-approval';

describe('approvalDetail', () => {
  it('omits the argument already shown in the summary', () => {
    expect(approvalDetail({ tool: 'file_read', arguments: { path: '~/a.md' } })).toBeNull();
    expect(approvalDetail({ tool: 'shell_exec', arguments: { command: 'ls', timeout: 5 } }))
      .toBe(JSON.stringify({ timeout: 5 }, null, 2));
  });

  it('shows the code a code_interpreter call would run', () => {
    expect(approvalDetail({ tool: 'code_interpreter', arguments: { code: 'print(1)' } })).toBe('print(1)');
  });

  it('previews file_write content, truncated', () => {
    const long = 'x'.repeat(700);
    const detail = approvalDetail({ tool: 'file_write', arguments: { path: 'p', content: long } });
    expect(detail?.length).toBe(601);
    expect(detail?.endsWith('…')).toBe(true);
  });

  it('shows all arguments for unknown tools', () => {
    expect(approvalDetail({ tool: 'custom', arguments: { a: 1 } })).toBe(JSON.stringify({ a: 1 }, null, 2));
  });
});

describe('secondsLeft', () => {
  it('counts down to the expiry and never goes negative', () => {
    expect(secondsLeft(100, 90_000)).toBe(10);
    expect(secondsLeft(100, 120_000)).toBe(0);
  });
});
