import { getBase } from './api';

/** Sample rate the server's wake-word model expects. */
export const WAKE_SAMPLE_RATE = 16000;

export type WakeServerEvent =
  | { type: 'ready'; wake_word: string }
  | { type: 'wake' }
  | { type: 'transcribing' }
  | { type: 'transcript'; text: string }
  | { type: 'timeout' }
  | { type: 'error'; detail: string };

export function buildWakeWsUrl(): string {
  const base = getBase();
  const url = new URL('/v1/speech/wake', base || window.location.origin);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

/**
 * Streaming box-filter resampler from the AudioContext rate to 16 kHz int16.
 *
 * Each output sample averages the input samples it covers, which doubles as a
 * cheap anti-aliasing low-pass. Leftover input is carried between calls so
 * chunk boundaries never drop or duplicate audio.
 */
export class PcmResampler {
  private readonly ratio: number;
  private leftover: Float32Array = new Float32Array(0);
  /** Absolute index of the next output sample. */
  private outIndex = 0;
  /** Absolute input index of ``leftover[0]``. */
  private inOffset = 0;

  constructor(inputRate: number, outputRate: number = WAKE_SAMPLE_RATE) {
    if (!(inputRate >= outputRate)) {
      throw new Error(`Cannot resample ${inputRate} Hz up to ${outputRate} Hz`);
    }
    this.ratio = inputRate / outputRate;
  }

  push(input: Float32Array): Int16Array<ArrayBuffer> {
    let buffer = input;
    if (this.leftover.length) {
      buffer = new Float32Array(this.leftover.length + input.length);
      buffer.set(this.leftover);
      buffer.set(input, this.leftover.length);
    }
    // Window boundaries come from absolute positions, so a non-integer ratio
    // (e.g. 44.1 kHz -> 16 kHz) never drifts across chunk boundaries.
    const values: number[] = [];
    for (;;) {
      const start = Math.floor(this.outIndex * this.ratio) - this.inOffset;
      const end = Math.max(
        start + 1,
        Math.floor((this.outIndex + 1) * this.ratio) - this.inOffset,
      );
      if (end > buffer.length) break;
      let sum = 0;
      for (let j = start; j < end; j++) sum += buffer[j];
      const value = Math.max(-1, Math.min(1, sum / (end - start)));
      values.push(value < 0 ? Math.round(value * 0x8000) : Math.round(value * 0x7fff));
      this.outIndex++;
    }
    const consumed = Math.floor(this.outIndex * this.ratio) - this.inOffset;
    this.leftover = buffer.slice(consumed);
    this.inOffset += consumed;
    return Int16Array.from(values);
  }
}

/** Short rising two-tone cue: "I heard you, go ahead". */
export function playWakeChime(ctx: AudioContext): void {
  const now = ctx.currentTime;
  [880, 1320].forEach((freq, i) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const start = now + i * 0.08;
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(0.15, start + 0.01);
    gain.gain.linearRampToValueAtTime(0, start + 0.07);
    osc.connect(gain).connect(ctx.destination);
    osc.start(start);
    osc.stop(start + 0.08);
  });
}
