import { useEffect, useRef, useState } from 'react';
import { buildWsProtocols } from '../lib/useAgentEvents';
import {
  PcmResampler,
  buildWakeWsUrl,
  playWakeChime,
  type WakeServerEvent,
} from '../lib/wake';

export type WakeStatus =
  | 'off'
  | 'connecting'
  | 'listening'
  | 'recording'
  | 'transcribing'
  | 'error';

interface UseWakeWordOptions {
  enabled: boolean;
  /** Stop streaming audio (e.g. while a reply generates or is spoken). */
  paused: boolean;
  onTranscript: (text: string) => void;
}

const MAX_BACKOFF_MS = 10_000;

/**
 * Hands-free "Hey Jarvis": while enabled, stream the microphone to the local
 * server's /v1/speech/wake socket, which detects the wake word, records until
 * the speaker pauses, and returns the transcript.
 *
 * Audio is withheld while ``paused`` so Jarvis cannot wake itself while it is
 * talking; on resume the server is told to drop any partial state.
 */
export function useWakeWord({ enabled, paused, onTranscript }: UseWakeWordOptions) {
  const [status, setStatus] = useState<WakeStatus>('off');
  const [error, setError] = useState<string | null>(null);
  const onTranscriptRef = useRef(onTranscript);
  onTranscriptRef.current = onTranscript;
  const pausedRef = useRef(paused);
  const busyRef = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const wasPaused = pausedRef.current;
    pausedRef.current = paused;
    const ws = wsRef.current;
    if (wasPaused && !paused && ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'reset' }));
      busyRef.current = false;
      setStatus((s) => (s === 'off' || s === 'error' ? s : 'listening'));
    }
  }, [paused]);

  useEffect(() => {
    if (!enabled) {
      setStatus('off');
      setError(null);
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('error');
      setError('Microphone not supported in this browser');
      return;
    }

    let cancelled = false;
    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;
    let processor: ScriptProcessorNode | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let backoff = 1000;

    const connect = () => {
      if (cancelled) return;
      setStatus('connecting');
      const ws = new WebSocket(buildWakeWsUrl(), buildWsProtocols());
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onmessage = (msg) => {
        let event: WakeServerEvent;
        try {
          event = JSON.parse(msg.data as string);
        } catch {
          return;
        }
        switch (event.type) {
          case 'ready':
            backoff = 1000;
            busyRef.current = false;
            setError(null);
            setStatus('listening');
            break;
          case 'wake':
            if (ctx) playWakeChime(ctx);
            setStatus('recording');
            break;
          case 'transcribing':
            busyRef.current = true;
            setStatus('transcribing');
            break;
          case 'transcript':
            busyRef.current = false;
            setStatus('listening');
            if (event.text.trim()) onTranscriptRef.current(event.text.trim());
            break;
          case 'timeout':
            setStatus('listening');
            break;
          case 'error':
            busyRef.current = false;
            setError(event.detail);
            break;
        }
      };

      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null;
        if (cancelled) return;
        setStatus('error');
        setError((e) => e ?? 'Lost connection to the wake-word service');
        retryTimer = setTimeout(connect, backoff);
        backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
      };
    };

    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      } catch {
        if (!cancelled) {
          setStatus('error');
          setError('Microphone access denied');
        }
        return;
      }
      if (cancelled) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      ctx = new AudioContext();
      // Autoplay policy can start the context suspended; resume now, and on
      // the next user gesture if the browser still refuses.
      const audioCtx = ctx;
      void audioCtx.resume().catch(() => {});
      if (audioCtx.state === 'suspended') {
        const resume = () => void audioCtx.resume().catch(() => {});
        window.addEventListener('pointerdown', resume, { once: true });
        window.addEventListener('keydown', resume, { once: true });
      }
      const resampler = new PcmResampler(ctx.sampleRate);
      const source = ctx.createMediaStreamSource(stream);
      // ScriptProcessorNode is deprecated but, unlike AudioWorklet, needs no
      // separately served module -- which keeps it working in the desktop
      // webview's CSP. The node must reach the destination to be scheduled;
      // it never writes its output buffer, so it plays silence.
      processor = ctx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => {
        const pcm = resampler.push(e.inputBuffer.getChannelData(0));
        const ws = wsRef.current;
        if (
          pausedRef.current
          || busyRef.current
          || !ws
          || ws.readyState !== WebSocket.OPEN
          || pcm.length === 0
        ) {
          return;
        }
        ws.send(pcm);
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      connect();
    })();

    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      wsRef.current?.close();
      wsRef.current = null;
      if (processor) processor.onaudioprocess = null;
      processor?.disconnect();
      stream?.getTracks().forEach((t) => t.stop());
      void ctx?.close();
      busyRef.current = false;
    };
  }, [enabled]);

  return { status, error };
}
