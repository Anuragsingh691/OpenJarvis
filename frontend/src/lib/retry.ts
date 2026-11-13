// Retry a startup request until the API answers. `jarvis gui` opens the
// browser once the frontend is up, which can be before the API server is
// ready, so a single failed fetch would otherwise leave the page empty
// (e.g. no models, "Pick a model first") until a manual reload.

const INITIAL_DELAY_MS = 500;
const MAX_DELAY_MS = 5000;

interface RetryOptions {
  isCancelled: () => boolean;
  sleep?: (ms: number) => Promise<void>;
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** Resolve with fn's result, or undefined if cancelled before it succeeds. */
export async function retryUntilSuccess<T>(
  fn: () => Promise<T>,
  { isCancelled, sleep = defaultSleep }: RetryOptions,
): Promise<T | undefined> {
  let delay = INITIAL_DELAY_MS;
  while (!isCancelled()) {
    try {
      return await fn();
    } catch {
      if (isCancelled()) return undefined;
      await sleep(delay);
      delay = Math.min(delay * 2, MAX_DELAY_MS);
    }
  }
  return undefined;
}
