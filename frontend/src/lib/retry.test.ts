import { describe, expect, it } from 'vitest';
import { retryUntilSuccess } from './retry';

const noSleep = (delays: number[]) => (ms: number) => {
  delays.push(ms);
  return Promise.resolve();
};

describe('retryUntilSuccess', () => {
  it('keeps retrying while the server is still starting, then returns the result', async () => {
    let calls = 0;
    const delays: number[] = [];
    const result = await retryUntilSuccess(
      async () => {
        calls++;
        if (calls < 4) throw new Error('ECONNREFUSED');
        return ['qwen3.5:9b'];
      },
      { isCancelled: () => false, sleep: noSleep(delays) },
    );
    expect(result).toEqual(['qwen3.5:9b']);
    expect(calls).toBe(4);
    expect(delays).toEqual([500, 1000, 2000]);
  });

  it('caps the backoff delay', async () => {
    let calls = 0;
    const delays: number[] = [];
    await retryUntilSuccess(
      async () => {
        if (++calls < 8) throw new Error('down');
        return 'ok';
      },
      { isCancelled: () => false, sleep: noSleep(delays) },
    );
    expect(Math.max(...delays)).toBe(5000);
  });

  it('stops and resolves undefined once cancelled', async () => {
    let calls = 0;
    let cancelled = false;
    const result = await retryUntilSuccess(
      async () => {
        calls++;
        cancelled = true;
        throw new Error('down');
      },
      { isCancelled: () => cancelled, sleep: () => Promise.resolve() },
    );
    expect(result).toBeUndefined();
    expect(calls).toBe(1);
  });
});
