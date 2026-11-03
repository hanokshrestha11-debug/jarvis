import os
import unittest
from unittest.mock import patch

import numpy as np

import levi_voice


class VoiceCommandTests(unittest.TestCase):
    def test_switch_commands_select_provider(self):
        examples = {
            "Switch to Claude": "anthropic",
            "change brain to Gemini": "gemini",
            "Use ChatGPT": "openai",
            "switch to local Ollama": "ollama",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                command = levi_voice._brain_command(text)
                self.assertIsNotNone(command)
                self.assertEqual(command[0], expected)

    def test_best_model_command_restores_automatic_routing(self):
        for text in (
            "use the best model",
            "switch to best model right now",
            "change brain to the best model",
        ):
            with self.subTest(text=text):
                command = levi_voice._brain_command(text)
                self.assertIsNotNone(command)
                self.assertEqual(command[0], "best")

    def test_regular_question_is_not_a_brain_command(self):
        self.assertIsNone(levi_voice._brain_command("Can you compare these models?"))

    def test_request_type_selects_a_task_route(self):
        self.assertEqual(
            levi_voice._task_provider_order("Help me debug this Python function"),
            ["anthropic", "openai", "gemini", "ollama"],
        )
        self.assertEqual(
            levi_voice._task_provider_order("Analyze and compare these strategies"),
            ["openai", "anthropic", "gemini", "ollama"],
        )
        self.assertEqual(
            levi_voice._task_provider_order("What time is it?"),
            ["gemini", "openai", "anthropic", "ollama"],
        )

    def test_auto_routing_falls_back_when_provider_fails(self):
        assistant = levi_voice.LeviAssistant()
        env = {
            "ANTHROPIC_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            levi_voice,
            "_chat_with_provider",
            side_effect=[levi_voice.ProviderError("offline"), "answer"],
        ) as chat:
            answer = assistant.respond("Help me debug this Python function")

        self.assertEqual(answer, "answer")
        self.assertEqual(
            [call.args[0] for call in chat.call_args_list],
            ["anthropic", "openai"],
        )
        self.assertEqual(assistant.history[-1], {"role": "assistant", "content": "answer"})

    def test_explicit_provider_does_not_silently_switch(self):
        assistant = levi_voice.LeviAssistant()
        assistant.selected_brain = "anthropic"
        with patch.object(
            levi_voice,
            "_chat_with_provider",
            side_effect=levi_voice.ProviderError("offline"),
        ) as chat:
            with self.assertRaises(levi_voice.ProviderError):
                assistant.respond("Answer this")
        chat.assert_called_once()
        self.assertEqual(chat.call_args.args[0], "anthropic")

    def test_audio_resampling_uses_expected_rate_ratio(self):
        source = np.ones(44100, dtype=np.float32)
        converted = levi_voice._resample(source, 44100, 16000)
        self.assertEqual(len(converted), 16000)
        self.assertTrue(np.allclose(converted, 1.0))

    def test_speech_capture_yields_utterance_and_pauses_microphone(self):
        class FakeStream:
            def __init__(self, **_kwargs):
                silence = np.zeros((4410, 1), dtype=np.float32)
                voice = np.full((4410, 1), 0.03, dtype=np.float32)
                self.frames = iter([silence] * 3 + [voice] * 5 + [silence] * 8)
                self.stop_count = 0
                self.start_count = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _frame_size):
                return next(self.frames), False

            def stop(self):
                self.stop_count += 1

            def start(self):
                self.start_count += 1

        fake_stream = FakeStream()
        with patch.dict(
            os.environ, {"LEVI_VOICE_THRESHOLD": "0.006"}, clear=True
        ), patch.object(
            levi_voice.sd, "InputStream", return_value=fake_stream
        ):
            utterances = levi_voice.speech_utterances(device=0)
            audio = next(utterances)
            utterances.close()

        self.assertEqual(len(audio), 25600)
        self.assertEqual(fake_stream.stop_count, 1)
        self.assertEqual(fake_stream.start_count, 0)

    def test_openai_responses_request(self):
        env = {"OPENAI_API_KEY": "test-key"}
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "OpenAI answer"}],
                }
            ]
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            levi_voice, "_post_json", return_value=response
        ) as post:
            answer = levi_voice._chat_with_provider(
                "openai", [{"role": "user", "content": "hello"}]
            )
        self.assertEqual(answer, "OpenAI answer")
        self.assertTrue(post.call_args.args[0].endswith("/responses"))
        self.assertEqual(post.call_args.args[1]["model"], "gpt-6-astra")
        self.assertEqual(post.call_args.args[2]["Authorization"], "Bearer test-key")

    def test_anthropic_messages_request(self):
        with patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True
        ), patch.object(
            levi_voice,
            "_post_json",
            return_value={"content": [{"type": "text", "text": "Claude answer"}]},
        ) as post:
            answer = levi_voice._chat_with_provider(
                "anthropic", [{"role": "user", "content": "hello"}]
            )
        self.assertEqual(answer, "Claude answer")
        self.assertEqual(post.call_args.args[2]["x-api-key"], "test-key")
        self.assertEqual(
            post.call_args.args[1]["model"], "claude-sonnet-5-5"
        )

    def test_gemini_generate_content_request(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), patch.object(
            levi_voice,
            "_post_json",
            return_value={
                "candidates": [
                    {"content": {"parts": [{"text": "Gemini answer"}]}}
                ]
            },
        ) as post:
            answer = levi_voice._chat_with_provider(
                "gemini", [{"role": "user", "content": "hello"}]
            )
        self.assertEqual(answer, "Gemini answer")
        self.assertEqual(post.call_args.args[2]["x-goog-api-key"], "test-key")
        self.assertIn("gemini-3.8-flash:generateContent", post.call_args.args[0])

    def test_ollama_chat_request(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            levi_voice,
            "_post_json",
            return_value={"message": {"content": "Local answer"}},
        ) as post:
            answer = levi_voice._chat_with_provider(
                "ollama", [{"role": "user", "content": "hello"}]
            )
        self.assertEqual(answer, "Local answer")
        self.assertEqual(
            post.call_args.args[0], "http://127.0.0.1:11434/api/chat"
        )
        self.assertEqual(post.call_args.args[1]["model"], "llama3.2")


if __name__ == "__main__":
    unittest.main()
