"""Continuous voice interaction and selectable AI providers for Levi."""

from __future__ import annotations

from collections import deque
import json
import logging
import os
import re
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import sounddevice as sd

log = logging.getLogger("levi")

AUDIO_RATE = 44100
TRANSCRIPTION_RATE = 16000
AUDIO_FRAME_SECONDS = 0.1
END_SILENCE_SECONDS = 0.8
MAX_UTTERANCE_SECONDS = 20.0
MIN_UTTERANCE_SECONDS = 0.3
PRE_ROLL_FRAMES = 3
DEFAULT_MODELS = {
    "openai": "gpt-6-astra",
    "anthropic": "claude-sonnet-5-5",
    "gemini": "gemini-3.8-flash",
    "ollama": "llama3.2",
}
PROVIDER_LABELS = {
    "openai": "OpenAI / ChatGPT",
    "anthropic": "Anthropic / Claude",
    "gemini": "Google / Gemini",
    "ollama": "local Ollama",
}
PROVIDER_ALIASES = {
    "openai": "openai",
    "chatgpt": "openai",
    "gpt": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
    "ollama": "ollama",
    "local": "ollama",
    "local ollama": "ollama",
}
SYSTEM_PROMPT = (
    "You are Levi, the personal AI assistant for Owner Hanuk Shrestha. "
    "Be helpful, clear, concise, and honest about uncertainty. Do not claim to have "
    "performed actions on the owner's computer unless the application confirms it."
)


class ProviderError(RuntimeError):
    """A provider is not configured, reachable, or able to answer."""


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise ProviderError(f"Connection failed: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderError(f"Provider returned invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ProviderError("Provider returned an unexpected response.")
    return result


def _required_text(value: Any, provider: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(f"{PROVIDER_LABELS[provider]} returned an empty answer.")
    return value.strip()


def _task_provider_order(prompt: str) -> list[str]:
    text = prompt.lower()
    coding_terms = (
        "code",
        "coding",
        "program",
        "python",
        "javascript",
        "typescript",
        "debug",
        "bug",
        "compile",
        "function",
        "class ",
        "script",
        "api",
    )
    reasoning_terms = (
        "analyze",
        "analysis",
        "compare",
        "reason",
        "reasoning",
        "strategy",
        "tradeoff",
        "trade-off",
        "math",
        "calculate",
        "prove",
        "complex",
        "plan",
        "evaluate",
    )
    if any(term in text for term in coding_terms):
        return ["anthropic", "openai", "gemini", "ollama"]
    if any(term in text for term in reasoning_terms):
        return ["openai", "anthropic", "gemini", "ollama"]
    return ["gemini", "openai", "anthropic", "ollama"]


def _brain_command(text: str) -> tuple[str, str] | None:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    normalized = " ".join(normalized.split())
    if re.fullmatch(
        r"(?:(?:hey levi|please) )?(?:(?:switch|change|set|use|select|choose) "
        r"(?:(?:the|your) )?(?:(?:ai )?(?:brain|model) )?(?:to )?)?"
        r"(?:the )?(?:best|best model|automatic|auto)(?: model| brain| model right now)?",
        normalized,
    ):
        return ("best", "Levi will choose a configured model based on each request.")

    match = re.fullmatch(
        r"(?:(?:hey levi|please) )?(?:switch|change|set|use|select|choose) "
        r"(?:(?:the|your) )?(?:(?:ai )?(?:brain|model) )?(?:to )?"
        r"(?:the )?(openai|chatgpt|gpt|anthropic|claude|gemini|google|ollama|local(?: ollama)?)"
        r"(?: model| brain)?",
        normalized,
    )
    if not match:
        return None
    provider = PROVIDER_ALIASES[match.group(1)]
    return (provider, f"Levi switched to {PROVIDER_LABELS[provider]}.")


def _provider_model(provider: str) -> str:
    return (
        os.environ.get(f"{provider.upper()}_MODEL")
        or DEFAULT_MODELS[provider]
    ).strip()


def _provider_configured(provider: str) -> bool:
    if provider == "ollama":
        return True
    return bool((os.environ.get(f"{provider.upper()}_API_KEY") or "").strip())


def _chat_with_provider(
    provider: str, history: list[dict[str, str]]
) -> str:
    model = _provider_model(provider)
    if provider == "openai":
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ProviderError("Set OPENAI_API_KEY in the local .env file.")
        data = _post_json(
            (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
            .rstrip("/")
            + "/responses",
            {
                "model": model,
                "instructions": SYSTEM_PROMPT,
                "input": history,
            },
            {"Authorization": f"Bearer {api_key}"},
        )
        content = data.get("output_text")
        if not isinstance(content, str):
            output = data.get("output")
            if not isinstance(output, list):
                raise ProviderError("OpenAI response did not contain a message.")
            content = " ".join(
                block["text"]
                for item in output
                if isinstance(item, dict) and item.get("type") == "message"
                for block in item.get("content", [])
                if isinstance(block, dict)
                and block.get("type") == "output_text"
                and isinstance(block.get("text"), str)
            )
        return _required_text(content, provider)

    if provider == "anthropic":
        api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not api_key:
            raise ProviderError("Set ANTHROPIC_API_KEY in the local .env file.")
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model": model,
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": history,
            },
            {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ProviderError("Anthropic response did not contain message content.")
        content = " ".join(
            block["text"]
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
        return _required_text(content, provider)

    if provider == "gemini":
        api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise ProviderError("Set GEMINI_API_KEY in the local .env file.")
        contents = [
            {
                "role": "user" if item["role"] == "user" else "model",
                "parts": [{"text": item["content"]}],
            }
            for item in history
        ]
        data = _post_json(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(model, safe='')}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": contents,
            },
            {"x-goog-api-key": api_key},
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini response did not contain a message.") from exc
        content = " ".join(
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        return _required_text(content, provider)

    if provider == "ollama":
        base_url = (
            os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
        ).strip().rstrip("/")
        data = _post_json(
            base_url + "/api/chat",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *history,
                ],
                "stream": False,
            },
            {},
        )
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError("Ollama response did not contain a message.") from exc
        return _required_text(content, provider)

    raise ProviderError(f"Unknown AI provider: {provider}")


class LeviAssistant:
    def __init__(self) -> None:
        self.selected_brain = "best"
        self.history: list[dict[str, str]] = []

    def respond(self, text: str) -> str:
        command = _brain_command(text)
        if command:
            self.selected_brain, response = command
            return response

        if self.selected_brain == "best":
            candidates = _task_provider_order(text)
            candidates = [p for p in candidates if _provider_configured(p)]
        else:
            candidates = [self.selected_brain]

        if not candidates:
            raise ProviderError(
                "No cloud API key is configured and Ollama is not available as a "
                "configured provider. Add API keys to .env or start Ollama."
            )

        conversation = [*self.history, {"role": "user", "content": text}]
        errors: list[str] = []
        for provider in candidates:
            try:
                answer = _chat_with_provider(provider, conversation)
            except ProviderError as exc:
                log.warning("%s failed: %s", PROVIDER_LABELS[provider], exc)
                errors.append(f"{PROVIDER_LABELS[provider]}: {exc}")
                if self.selected_brain != "best":
                    break
                continue
            self.history.extend(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": answer},
                ]
            )
            self.history = self.history[-20:]
            return answer

        detail = "; ".join(errors) or "No configured providers were available."
        raise ProviderError(f"No AI provider could answer. {detail}")


def _resample(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    output_length = round(len(audio) * target_rate / sample_rate)
    if output_length < 1:
        return np.empty(0, dtype=np.float32)
    old_positions = np.arange(len(audio), dtype=np.float64)
    new_positions = np.linspace(0, len(audio) - 1, output_length)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def speech_utterances(device: int) -> Iterator[np.ndarray]:
    """Yield speech audio, pausing the mic while a yielded utterance is handled."""
    frame_size = round(AUDIO_RATE * AUDIO_FRAME_SECONDS)
    pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_FRAMES)
    speech: list[np.ndarray] = []
    noise_floor = 0.001
    silence_frames = 0
    active = False
    min_silence_frames = round(END_SILENCE_SECONDS / AUDIO_FRAME_SECONDS)
    max_speech_frames = round(MAX_UTTERANCE_SECONDS / AUDIO_FRAME_SECONDS)
    try:
        threshold_floor = float(os.environ.get("LEVI_VOICE_THRESHOLD", "0.006"))
    except ValueError as exc:
        raise ValueError("LEVI_VOICE_THRESHOLD must be a positive number.") from exc
    if threshold_floor <= 0:
        raise ValueError("LEVI_VOICE_THRESHOLD must be a positive number.")

    with sd.InputStream(
        device=device,
        samplerate=AUDIO_RATE,
        channels=1,
        dtype="float32",
        blocksize=frame_size,
    ) as stream:
        while True:
            data, overflowed = stream.read(frame_size)
            if overflowed:
                log.warning("Microphone input overflowed; speech may be incomplete.")
            frame = data[:, 0].copy()
            level = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            threshold = max(threshold_floor, noise_floor * 3.0)

            if not active:
                if level >= threshold:
                    active = True
                    speech = [*pre_roll, frame]
                    silence_frames = 0
                else:
                    noise_floor = max(1e-6, noise_floor * 0.98 + level * 0.02)
                    pre_roll.append(frame)
                continue

            speech.append(frame)
            if level < threshold:
                silence_frames += 1
            else:
                silence_frames = 0

            if (
                silence_frames >= min_silence_frames
                or len(speech) >= max_speech_frames
            ):
                audio = np.concatenate(speech)
                duration = len(audio) / AUDIO_RATE
                active = False
                speech = []
                silence_frames = 0
                pre_roll.clear()
                if duration < MIN_UTTERANCE_SECONDS:
                    continue
                stream.stop()
                yield _resample(audio, AUDIO_RATE, TRANSCRIPTION_RATE)
                stream.start()


def _transcribe(model: Any, audio: np.ndarray) -> str:
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=3,
        vad_filter=True,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def run_voice_assistant(
    *,
    input_device: int,
    welcome_enabled: bool,
    welcome_phrase: str,
) -> int:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.error("Install voice dependencies with: python -m pip install -r requirements.txt")
        return 1
    try:
        import pyttsx3
    except ImportError:
        log.error("Install voice dependencies with: python -m pip install -r requirements.txt")
        return 1

    model_name = (os.environ.get("WHISPER_MODEL") or "base").strip()
    try:
        whisper = WhisperModel(model_name, device="cpu", compute_type="int8")
        speaker = pyttsx3.init()
    except (OSError, RuntimeError, ValueError) as exc:
        log.error("Could not initialize Levi's local voice services: %s", exc)
        return 1

    assistant = LeviAssistant()

    def speak(text: str) -> None:
        print(f"Levi: {text}", flush=True)
        speaker.say(text)
        speaker.runAndWait()

    print(
        "Levi is listening continuously. Say “switch to Claude/Gemini/ChatGPT/Ollama” "
        "or “use the best model” to change brains. Press Ctrl+C to stop.",
        flush=True,
    )
    if welcome_enabled and welcome_phrase.strip():
        speak(welcome_phrase.strip())

    try:
        for audio in speech_utterances(input_device):
            try:
                text = _transcribe(whisper, audio)
            except (OSError, RuntimeError, ValueError) as exc:
                log.error("Local speech transcription failed: %s", exc)
                speak("I could not transcribe that. Please try again.")
                continue
            if not text:
                continue
            print(f"You: {text}", flush=True)
            command_text = " ".join(
                re.sub(r"[^a-z ]+", " ", text.lower()).split()
            )
            if re.fullmatch(
                r"(?:hey levi )?(?:stop listening|exit|quit|goodbye)",
                command_text,
            ):
                speak("Goodbye, Owner.")
                return 0
            try:
                speak(assistant.respond(text))
            except ProviderError as exc:
                log.error("%s", exc)
                speak("I could not get an answer. Check the provider setup and try again.")
    except KeyboardInterrupt:
        log.info("Levi stopped.")
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    except sd.PortAudioError as exc:
        log.error("Microphone error: %s", exc)
        log.error("Check Windows microphone permissions and the selected input device.")
        return 1
    return 0
