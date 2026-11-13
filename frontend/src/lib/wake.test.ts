import { afterEach, describe, expect, it, vi } from 'vitest';
import { PcmResampler, WAKE_SAMPLE_RATE, buildWakeWsUrl } from './wake';

describe('PcmResampler', () => {
  it('downsamples 48 kHz to 16 kHz by averaging groups of three', () => {
    const r = new PcmResampler(48000);
    const out = r.push(new Float32Array([0.3, 0.3, 0.3, -0.6, -0.6, -0.6]));
    expect(Array.from(out)).toEqual([Math.round(0.3 * 0x7fff), Math.round(-0.6 * 0x8000)]);
  });

  it('carries leftover samples across chunks without loss', () => {
    const whole = new PcmResampler(44100);
    const split = new PcmResampler(44100);
    const input = Float32Array.from({ length: 4410 }, (_, i) => Math.sin(i / 7));
    const expected = whole.push(input);
    const a = split.push(input.subarray(0, 1000));
    const b = split.push(input.subarray(1000));
    expect(a.length + b.length).toBe(expected.length);
    expect([...a, ...b]).toEqual(Array.from(expected));
    expect(expected.length).toBe(Math.floor(4410 / (44100 / WAKE_SAMPLE_RATE)));
  });

  it('clamps out-of-range input to int16', () => {
    const out = new PcmResampler(16000).push(new Float32Array([2, -2]));
    expect(Array.from(out)).toEqual([32767, -32768]);
  });

  it('refuses to upsample', () => {
    expect(() => new PcmResampler(8000)).toThrow();
  });
});

describe('buildWakeWsUrl', () => {
  afterEach(() => vi.unstubAllGlobals());

  const stubPage = (origin: string, settings?: object) => {
    vi.stubGlobal('window', { location: { origin } });
    vi.stubGlobal('localStorage', {
      getItem: () => (settings ? JSON.stringify(settings) : null),
    });
  };

  it('targets the wake socket on the page origin by default', () => {
    stubPage('http://localhost:5173');
    expect(buildWakeWsUrl()).toBe('ws://localhost:5173/v1/speech/wake');
  });

  it('uses wss for an https API base from settings', () => {
    stubPage('http://localhost:5173', { apiUrl: 'https://jarvis.example:8443/' });
    expect(buildWakeWsUrl()).toBe('wss://jarvis.example:8443/v1/speech/wake');
  });
});
