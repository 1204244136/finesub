"""Spent so far by the edit judge: text input (counter tokens x attempts, per reply file) and audio seconds.

    python tools/diarization_explore/edit_judge_usage.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edit_judge as ej  # noqa: E402


def main():
    from finesub.llm.token_budget import default_token_counter
    count = default_token_counter().count_text
    text = audio = calls = attempts = 0
    for task in ej.TASKS:
        for arm in ej.ARMS:
            prompts = {c[0]: (count(c[1]) + count(c[2]), c[4]) for c in ej.calls(task, arm)}
            for folder in sorted((ej.OUT / f"{task}-{arm}").glob("r*")):
                t = a = n = k = 0
                for path in folder.glob("*.json"):
                    key = path.name.split(".")[0]
                    tries = json.loads(path.read_text(encoding="utf-8")).get("tries", 1)
                    tokens, seconds = prompts[key]
                    t, a, n, k = t + tokens * tries, a + seconds * tries, n + 1, k + tries
                print(f"{task}-{arm} {folder.name}: {n} windows, {k} attempts, text {t}, audio {a:.0f} s")
                text, audio, calls, attempts = text + t, audio + a, calls + n, attempts + k
    print(f"total: {calls} windows, {attempts} attempts, text input {text}, audio {audio:.0f} s ≈ {audio * 32:.0f} tokens")


if __name__ == "__main__":
    main()
