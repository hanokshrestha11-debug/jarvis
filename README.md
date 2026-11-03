# Levi — Personal AI Assistant

Levi is a voice assistant created for its owner, **Hanuk Shrestha**. It continuously listens through the selected microphone, transcribes speech locally, answers with a chosen AI provider, and speaks replies using the computer's local voice.

## Setup

Levi's local Whisper dependency currently requires **Python 3.14**. Install Python 3.14, then use the Windows Python launcher (`py`) to create the environment and run Levi:

```powershell
py -3.14 -m pip install -r requirements.txt
py -3.14 jarvis.py
```

In VS Code, use **Python: Select Interpreter** and choose Python 3.14 for this workspace.

If you do not already have a `.env`, copy `.env.example` to `.env`; otherwise, add the settings you need to your existing file so you do not overwrite its keys. Keep `.env` private; never commit API keys. Optionally install [Ollama](https://ollama.com), start it, and download a model, for example `ollama pull llama3.2`.

The first run downloads the configured local Whisper speech-recognition model. Later transcriptions run locally. Levi needs internet access to call a selected cloud AI; Ollama requests stay on your machine.

## Voice commands

- Ask questions naturally; speech is transcribed locally and sent to the selected brain.
- Say **“switch to Claude,” “switch to Gemini,” “switch to ChatGPT,”** or **“switch to Ollama”** to use that provider for this session.
- Say **“use the best model”** or **“switch to best model”** to return to automatic task-based routing.
- Say **“stop listening”** or press **Ctrl+C** to exit.

Automatic routing uses a simple task category (coding, reasoning, or general question), then tries configured providers in order until one answers. This is a practical fallback policy, not a live benchmark or guarantee that one vendor is objectively best. Provider choice and conversation history last only for the current run.

## Configuration

Set keys and optional model overrides in `.env`:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI / ChatGPT API key. |
| `OPENAI_MODEL` | OpenAI model (default: `gpt-6-astra`). |
| `ANTHROPIC_API_KEY` | Anthropic / Claude API key. |
| `ANTHROPIC_MODEL` | Anthropic model (default: `claude-sonnet-5-5`). |
| `GEMINI_API_KEY` | Google AI API key. |
| `GEMINI_MODEL` | Gemini model (default: `gemini-3.8-flash`). |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://127.0.0.1:11434`). |
| `OLLAMA_MODEL` | Installed Ollama model (default: `llama3.2`). |
| `WHISPER_MODEL` | Local transcription model (default: `base`; larger models may be more accurate but slower). |
| `LEVI_VOICE_THRESHOLD` | Microphone speech-detection floor (default: `0.006`; increase to ignore room noise, decrease if speech is missed). |
| Preset greeting | Edit `JARVIS_WELCOME_ENABLED` or `JARVIS_WELCOME_PHRASE` near the top of `jarvis.py`. |

You only need keys for the cloud providers you use. API access and usage charges are separate from consumer ChatGPT, Claude, or Gemini subscriptions. When automatic routing reaches an unconfigured cloud provider, it skips it. If all cloud services fail, Levi tries local Ollama. A provider explicitly selected by voice does not silently switch to another provider.

The microphone stays on the PC and Whisper transcribes audio locally; raw audio is not sent to cloud AI providers. Transcripts and conversation context are sent to whichever cloud provider answers. To keep prompts local, say **“switch to Ollama”**; Ollama must be running with the configured model installed.

## Troubleshooting

- **No response to speech:** Check the selected Windows microphone and its privacy permission. Adjust `LEVI_VOICE_THRESHOLD` in `.env`.
- **Whisper model download fails:** Check internet access on first run; the selected model is cached for later local use.
- **Provider errors:** Check the provider's API key, model name, account access, and billing. For Ollama, start its service and pull the configured model.
- **No spoken replies:** Levi uses the operating system's installed speech voice; check Windows sound output and speech settings.
- **Stop:** Press **Ctrl+C** or say **“stop listening.”**
