import os
import time
import json
import requests

MODEL = "google/gemini-2.0-flash-001"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "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"

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY environment variable is not set. Set it before running."
    )

PERSONA_SYSTEM_PROMPT = (
    "You generate closed-factual question and answer pairs. "
    "Each response must be exactly two lines in plain text: the first line is a concise question, "
    "the second line is the correct answer. Do not add labels like 'Question:' or 'Answer:'. "
    "Use safe, neutral topics (science, geography, history, math, general knowledge). "
    "Avoid sensitive, explicit, hateful, or unsafe content. Keep each line brief."
)

SEED_USER_PROMPT = (
    "Generate one new factual question and its answer. Output exactly two lines: "
    "first the question, second the answer. No labels or extra text."
)

messages = [
    {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
    {"role": "user", "content": SEED_USER_PROMPT},
]

OUTPUT_PATH = "qa.txt"
RECENT_MEMORY_LIMIT = 300
RECENT_IN_PROMPT = 50


def chat_once(chat_history):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "custom-llm qa loop",
    }
    payload = {
        "model": MODEL,
        "messages": chat_history,
        "temperature": 0.7,

    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def append_to_file(text):
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n\n")


def normalize_pair_output(raw_text):
    # Remove common labels and keep only the first two non-empty lines
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines[:2]:
        lower = ln.lower()
        if lower.startswith("question:") or lower.startswith("q:"):
            ln = ln.split(":", 1)[1].strip()
        if lower.startswith("answer:") or lower.startswith("a:"):
            ln = ln.split(":", 1)[1].strip()
        cleaned.append(ln)
    if not cleaned:
        return raw_text.strip()
    return "\n".join(cleaned)


def extract_question_line(pair_text: str) -> str:
    lines = [ln.strip() for ln in pair_text.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def normalize_question(q: str) -> str:
    return " ".join(q.lower().split())


def load_recent_questions_from_file(path: str, max_items: int = 300):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return []
    blocks = [blk.strip() for blk in text.split("\n\n") if blk.strip()]
    qs = []
    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        if lines:
            qs.append(lines[0])
    return qs[-max_items:]


def build_user_prompt(recent_questions):
    prompt = (
        "Generate another new factual question and its answer on a different topic. "
        "Do not repeat earlier pairs in this chat. Output exactly two lines: "
        "first the question, second the answer. No labels or extra text."
    )
    if recent_questions:
        last = recent_questions[-RECENT_IN_PROMPT:]
        bullets = "\n".join(f"- {q}" for q in last)
        prompt += (
            "\nAvoid repeating any of these recent questions (do not paraphrase them):\n"
            f"{bullets}"
        )
    return prompt


def shrink_context(full_messages, max_chars=6000):
    # Preserve the system prompt, and keep only the most recent user+assistant cycle(s)
    kept = [full_messages[0]]
    rest = full_messages[1:]
    while True:
        serialized = json.dumps(kept + rest, ensure_ascii=False)
        if len(serialized) <= max_chars:
            break
        if len(rest) > 2:
            rest = rest[-2:]
        else:
            break
    return kept + rest


def main():
    print("Starting Q->A generation loop via OpenRouter. Press Ctrl+C to stop.")
    iteration = 0
    recent_questions = load_recent_questions_from_file(OUTPUT_PATH, RECENT_MEMORY_LIMIT)
    recent_q_set = {normalize_question(q) for q in recent_questions}
    # Seed the chat with recent questions to avoid duplication on first turn
    messages.append({
        "role": "user",
        "content": build_user_prompt(recent_questions),
    })
    while True:
        try:
            iteration += 1
            safe_messages = shrink_context(messages)
            content = chat_once(safe_messages)
            content = normalize_pair_output(content)
            q_line = extract_question_line(content)
            q_key = normalize_question(q_line) if q_line else ""
            if q_key and q_key in recent_q_set:
                print(f"[{iteration}] Duplicate question detected. Retrying...", flush=True)
                messages.append({
                    "role": "user",
                    "content": build_user_prompt(recent_questions),
                })
                time.sleep(1)
                continue
            print(f"[{iteration}]\n{content}\n", flush=True)
            append_to_file(content)
            if q_line:
                recent_questions.append(q_line)
                recent_q_set.add(q_key)
                if len(recent_questions) > RECENT_MEMORY_LIMIT:
                    removed = recent_questions.pop(0)
                    rem_key = normalize_question(removed)
                    if rem_key in recent_q_set:
                        try:
                            recent_q_set.remove(rem_key)
                        except KeyError:
                            pass
            # Feed back the output and request a new pair with context of recent questions
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": build_user_prompt(recent_questions),
            })
            time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except requests.HTTPError as http_err:
            status = getattr(http_err.response, "status_code", None)
            print(f"HTTP error {status}: {http_err}. Retrying in 10s...", flush=True)
            time.sleep(10)
        except Exception as err:
            print(f"Error: {err}. Retrying in 5s...", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()








