from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import utc_now, write_json, write_text


@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    raw: dict[str, Any]


class OpenAICompatibleClient:
    def __init__(
        self,
        model_url: str,
        model_name: str,
        temperature: float,
        seed: int,
        max_completion_tokens: int,
        dry_run: bool,
    ) -> None:
        self.model_url = model_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.max_completion_tokens = max_completion_tokens
        self.dry_run = dry_run

    def complete(self, prompt: str, condition: str, log_dir: Path, call_name: str) -> ModelResponse:
        if self.dry_run:
            text = self._dry_completion(prompt, condition, call_name)
            response = ModelResponse(
                text=text,
                prompt_tokens=approx_tokens(prompt),
                completion_tokens=approx_tokens(text),
                raw={"dry_run": True, "content": text},
            )
        else:
            response = self._remote_completion(prompt)
        self._log_call(prompt, response, condition, log_dir, call_name)
        return response

    def _remote_completion(self, prompt: str) -> ModelResponse:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_completion_tokens,
            "seed": self.seed,
        }
        request = urllib.request.Request(
            f"{self.model_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as handle:
                raw = json.loads(handle.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Model call failed against {self.model_url}. Start an OpenAI-compatible local server "
                f"or rerun with --dry_run. Error: {exc}"
            ) from exc
        text = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage", {})
        return ModelResponse(
            text=text,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=raw,
        )

    def _dry_completion(self, prompt: str, condition: str, call_name: str) -> str:
        if "decision_change" in call_name:
            return json.dumps(
                {
                    "decision_changed": True,
                    "change_type": "added_ablation",
                    "reason": "Dry-run critique recommends a smaller controlled follow-up.",
                    "revised_experiment_plan": "Run a bounded ablation that changes one world-model setting and compares the primary metric.",
                },
                indent=2,
            )
        return (
            "1. Strongest concern: the current interpretation may over-credit one noisy metric.\n"
            "2. Missing control, baseline, or ablation: add a one-variable ablation against the first experiment.\n"
            "3. Metric or evaluation risk: report primary score plus runtime and failure status.\n"
            "4. Implementation/debug risk: verify the changed path is inside allowed_files.\n"
            "5. Suggested next experiment: keep compute fixed and run a smaller controlled variant.\n"
            "6. Falsification: no improvement or a failed run under the same budget.\n"
            "7. Run/change recommendation: change the proposed next experiment to the controlled ablation.\n"
            f"\nCondition context: {condition}. Prompt chars: {len(prompt)}.\n"
        )

    def _log_call(
        self,
        prompt: str,
        response: ModelResponse,
        condition: str,
        log_dir: Path,
        call_name: str,
    ) -> None:
        prompts_dir = log_dir / "prompts"
        completions_dir = log_dir / "completions"
        write_text(prompts_dir / f"{call_name}.txt", prompt)
        write_text(completions_dir / f"{call_name}.txt", response.text)
        write_json(
            log_dir / "model_calls" / f"{call_name}.json",
            {
                "timestamp": utc_now(),
                "condition": condition,
                "model_endpoint": self.model_url,
                "model_name": self.model_name,
                "temperature": self.temperature,
                "seed": self.seed,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            },
        )


def approx_tokens(text: str) -> int:
    return max(1, len(text.split()) + len(text) // 20)
