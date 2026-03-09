import os
import json
import sys
import time
import logging
import traceback
import re
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# =====================================================
# 🔧 SETUP
# =====================================================

load_dotenv(dotenv_path=Path(".env"))
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

if not GROQ_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")
if not OPENROUTER_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

groq_client = Groq(api_key=GROQ_KEY)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# =====================================================
# ⚙️ CONSTANTS
# =====================================================

BRAIN_MODEL        = "llama-3.1-8b-instant"
GROQ_STRONG_MODEL  = "llama-3.3-70b-versatile"
GROQ_FAST_MODEL    = "llama-3.1-8b-instant"

QUALITY_THRESHOLD  = 7    # min score (1-10) to accept an answer
MAX_RETRIES        = 3    # max retries per sub-question before giving up
MEMORY_PATH        = Path("./memory.json")
MAX_MEMORY_CONTEXT = 6    # last N turns sent to brain as context

# Known valid Groq model IDs for validation
KNOWN_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# Escalation path when answer quality is low
ESCALATION_ORDER = [
    ("openrouter", "openai/gpt-4o"),
    ("openrouter", "anthropic/claude-3.5-sonnet"),
    ("openrouter", "deepseek/deepseek-r1"),
    ("groq",       GROQ_STRONG_MODEL),
]


# =====================================================
# 💾 MEMORY MANAGER
# =====================================================

def load_memory() -> list:
    if MEMORY_PATH.exists():
        try:
            with open(MEMORY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_memory(memory: list):
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)


def add_to_memory(memory: list, role: str, content: str):
    memory.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    save_memory(memory)


def format_memory_context(memory: list) -> str:
    recent = memory[-MAX_MEMORY_CONTEXT:] if len(memory) > MAX_MEMORY_CONTEXT else memory
    if not recent:
        return "No prior conversation history."
    lines = []
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Agent"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


# =====================================================
# 🛡️ SAFE JSON PARSER
# =====================================================

def safe_json_parse(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


# =====================================================
# 🧠 STEP 1 — DECOMPOSE QUERY
# =====================================================

def decompose_query(query: str, memory: list) -> list:
    context = format_memory_context(memory)

    prompt = f"""
You are a smart planning brain. Break the user query into clear sub-questions that can be answered independently.

Prior conversation context:
{context}

User Query:
\"\"\"{query}\"\"\"

Rules:
- If the query is simple or conversational, return just 1 sub-question (the query itself).
- If complex or multi-part, break it into 2-4 focused sub-questions.
- Each sub-question must be fully self-contained.

Return ONLY a valid JSON array of strings, no markdown, no explanation:
["sub-question 1", "sub-question 2", ...]
"""

    response = groq_client.chat.completions.create(
        model=BRAIN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
        if isinstance(result, list) and all(isinstance(s, str) for s in result):
            return result
    except Exception:
        pass

    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except Exception:
            pass

    return [query]


# =====================================================
# 🧠 STEP 2 — BRAIN FREELY REASONS & ROUTES
# =====================================================

def route_subquestion(subq: str) -> dict:
    """
    Brain classifies query type, reasons freely about it,
    then picks ANY executor + model it deems best.
    No hardcoded thresholds — pure brain judgment.
    """

    prompt = f"""
You are an intelligent AI routing brain. Your job is to reason about this sub-question and decide
which AI model and executor will handle it best.

Sub-question:
\"\"\"{subq}\"\"\"

Think step by step:
1. What TYPE of query is this? (e.g. math, coding, reasoning, factual lookup, creative writing,
   scientific explanation, long-form writing, conversational, summarization, translation, etc.)
2. What does answering it well require? (speed, depth, long context, code execution, specialized knowledge, creativity?)
3. Should it go to Groq (fast inference, good for simple/moderate tasks) or OpenRouter
   (access to the most powerful models like GPT-4o, Claude 3.5, DeepSeek R1, Gemini, etc.)?
4. Which exact model is the absolute best fit for this specific query?

Available Groq models (fast, free):
- llama-3.1-8b-instant     → simple, fast, conversational
- llama-3.3-70b-versatile  → moderate to complex, strong reasoning
- mixtral-8x7b-32768       → good for multilingual, long context
- gemma2-9b-it             → efficient, good for structured tasks

Available OpenRouter models (powerful, costs credits):
- openai/gpt-4o                     → best general intelligence, analysis
- anthropic/claude-3.5-sonnet       → best for code, writing, nuanced reasoning
- deepseek/deepseek-r1              → best for math, logic, step-by-step
- google/gemini-pro-1.5             → best for very long context, documents
- mistralai/mixtral-8x22b-instruct  → powerful open model, fast
- meta-llama/llama-3.1-405b         → largest open model, deep reasoning
- (you may also pick any other model you know on openrouter.ai)

Return ONLY valid JSON, no markdown:
{{
  "query_type": "<type of query>",
  "brain_reasoning": "<2-3 sentences explaining your routing logic>",
  "chosen_executor": "groq" or "openrouter",
  "chosen_model": "<exact model id>",
  "confidence": <int 1-10>
}}
"""

    response = groq_client.chat.completions.create(
        model=BRAIN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    parsed = safe_json_parse(response.choices[0].message.content)

    if not parsed:
        return {
            "query_type": "unknown",
            "brain_reasoning": "Could not parse routing decision — using safe fallback.",
            "chosen_executor": "groq",
            "chosen_model": GROQ_FAST_MODEL,
            "confidence": 5
        }

    # ── Validate: if brain picked an unknown Groq model, fallback ──
    executor = parsed.get("chosen_executor", "groq")
    model    = parsed.get("chosen_model", GROQ_FAST_MODEL)

    if executor == "groq" and model not in KNOWN_GROQ_MODELS:
        print(f"        ⚠️  Brain picked unknown Groq model '{model}' → using {GROQ_STRONG_MODEL}")
        parsed["chosen_model"] = GROQ_STRONG_MODEL
        parsed["brain_reasoning"] += f" [auto-corrected: '{model}' → {GROQ_STRONG_MODEL}]"

    return parsed


# =====================================================
# ⚡ EXECUTORS
# =====================================================

def execute_groq(query: str, model: str, context: str = "") -> str:
    messages = []
    if context:
        messages.append({"role": "system", "content": f"Conversation context:\n{context}"})
    messages.append({"role": "user", "content": query})

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "decommissioned" in err or "not_found" in err or "not found" in err.lower():
            print(f"        ⚠️  Groq model '{model}' unavailable → falling back to {GROQ_STRONG_MODEL}")
            response = groq_client.chat.completions.create(
                model=GROQ_STRONG_MODEL,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        raise


def execute_openrouter(query: str, model: str, context: str = "") -> str:
    messages = []
    if context:
        messages.append({"role": "system", "content": f"Conversation context:\n{context}"})
    messages.append({"role": "user", "content": query})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/autonomous-llm-orchestrator",
        "X-Title": "Autonomous LLM Orchestrator"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code == 402:
        print(f"        💳 OpenRouter out of credits → falling back to Groq ({GROQ_STRONG_MODEL})")
        return execute_groq(query, GROQ_STRONG_MODEL, context)

    if response.status_code == 404:
        print(f"        ⚠️  OpenRouter model '{model}' not found → falling back to Groq ({GROQ_STRONG_MODEL})")
        return execute_groq(query, GROQ_STRONG_MODEL, context)

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


def execute(subq: str, executor: str, model: str, context: str = "") -> str:
    if executor == "groq":
        return execute_groq(subq, model, context)
    else:
        return execute_openrouter(subq, model, context)


# =====================================================
# 🧠 STEP 3 — SELF-REFLECT ON ANSWER
# =====================================================

def reflect_on_answer(subq: str, answer: str) -> dict:
    prompt = f"""
You are a critical evaluator. Score this answer strictly and honestly.

Question: \"\"\"{subq}\"\"\"
Answer: \"\"\"{answer}\"\"\"

Return ONLY valid JSON, no markdown:
{{
  "score": <int 1-10>,
  "reason": "<one sentence explaining the score>",
  "is_acceptable": <true if score >= {QUALITY_THRESHOLD} else false>
}}
"""

    response = groq_client.chat.completions.create(
        model=BRAIN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    parsed = safe_json_parse(response.choices[0].message.content)

    if not parsed:
        return {"score": 7, "reason": "Could not evaluate.", "is_acceptable": True}

    parsed["is_acceptable"] = parsed.get("score", 0) >= QUALITY_THRESHOLD
    return parsed


# =====================================================
# 🧠 STEP 4 — SYNTHESIZE ALL SUB-ANSWERS
# =====================================================

def synthesize_answers(original_query: str, parts: list) -> str:
    parts_text = ""
    for i, p in enumerate(parts, 1):
        parts_text += f"\nSub-question {i}: {p['subq']}\nAnswer {i}: {p['answer']}\n"

    prompt = f"""
You are a synthesis brain. Combine these sub-answers into one clear, complete, final answer.

Original Query: \"\"\"{original_query}\"\"\"

Sub-answers:
{parts_text}

Rules:
- Write as one unified answer, not a list of parts
- Be concise but complete
- Merge overlapping content naturally
- Do not mention sub-questions in your answer
"""

    response = groq_client.chat.completions.create(
        model=GROQ_STRONG_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# =====================================================
# 🖨️ PRINT HELPERS
# =====================================================

def print_header():
    print("\n" + "═" * 62)
    print("      🤖  AUTONOMOUS LLM ORCHESTRATOR  |  Agentic Mode")
    print("═" * 62)
    print("  Type your question. Commands: 'memory', 'exit', 'quit'")
    print("═" * 62 + "\n")


def print_routing(idx: int, total: int, subq: str, route: dict):
    executor   = route.get("chosen_executor", "groq")
    model      = route.get("chosen_model", "unknown")
    qtype      = route.get("query_type", "unknown")
    reasoning  = route.get("brain_reasoning", "")
    confidence = route.get("confidence", "?")
    icon       = "⚡ Groq" if executor == "groq" else "🌐 OpenRouter"

    print(f"  [{idx}/{total}] {icon} → {model}")
    print(f"        query type : {qtype}  |  confidence: {confidence}/10")
    print(f"        reasoning  : {reasoning}")
    print(f"        sub-q      : {subq[:80]}{'...' if len(subq) > 80 else ''}")


def print_reflection(score: int, reason: str, accepted: bool, attempt: int):
    icon   = "✅" if accepted else "⚠️ "
    status = "accepted" if accepted else f"retrying (attempt {attempt + 1})"
    print(f"        {icon} Score {score}/10 — {status}")
    print(f"           reason : {reason}")


def print_divider():
    print("  " + "─" * 58)


# =====================================================
# 🚀 CORE AGENT — processes one query
# =====================================================

def run_agent(query: str, memory: list):
    start = time.time()

    # ── Step 1: Decompose ──────────────────────────────────
    print("\n🧠 Brain decomposing query...")
    subquestions = decompose_query(query, memory)
    n = len(subquestions)

    if n == 1:
        print(f"  → Simple query, no decomposition needed.\n")
    else:
        print(f"  → Decomposed into {n} sub-questions.\n")

    # ── Step 2–3: Route + Execute + Reflect each part ──────
    parts   = []
    context = format_memory_context(memory)

    for idx, subq in enumerate(subquestions, 1):
        if n > 1:
            print_divider()

        # Brain freely reasons and picks model
        route    = route_subquestion(subq)
        print_routing(idx, n, subq, route)

        executor = route["chosen_executor"]
        model    = route["chosen_model"]
        answer   = None
        accepted = False

        for attempt in range(MAX_RETRIES):
            try:
                answer = execute(subq, executor, model, context)
            except Exception as e:
                print(f"        ❌ Execution error: {e}")
                answer = ""

            reflection = reflect_on_answer(subq, answer)
            score      = reflection.get("score", 0)
            reason     = reflection.get("reason", "")
            accepted   = reflection.get("is_acceptable", False)

            print_reflection(score, reason, accepted, attempt)

            if accepted:
                break

            # Escalate through fallback chain
            if attempt < len(ESCALATION_ORDER):
                next_executor, next_model = ESCALATION_ORDER[attempt]
                executor = next_executor
                model    = next_model
                print(f"        🔀 Escalating to → {next_executor.upper()} / {next_model}")
            else:
                print(f"        ⛔ Max escalation reached, using best available answer.")
                break

        parts.append({"subq": subq, "answer": answer or "No answer generated."})

    # ── Step 4: Synthesize ─────────────────────────────────
    if n > 1:
        print_divider()
        print("\n🔗 Synthesizing final answer from all parts...")
        final_answer = synthesize_answers(query, parts)
    else:
        final_answer = parts[0]["answer"]

    elapsed = time.time() - start

    # ── Print final answer ─────────────────────────────────
    print("\n" + "═" * 62)
    print("✅  FINAL ANSWER")
    print("═" * 62)
    print(final_answer)
    print("─" * 62)
    print(f"⏱️  Completed in {elapsed:.2f}s\n")

    # ── Save to memory ─────────────────────────────────────
    add_to_memory(memory, "user", query)
    add_to_memory(memory, "assistant", final_answer)

    return final_answer


# =====================================================
# 🔁 CHAT LOOP
# =====================================================

def chat_loop():
    print_header()
    memory = load_memory()

    if memory:
        print(f"📂 Loaded {len(memory)} messages from memory.json\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        if query.lower() == "memory":
            print("\n📜 Conversation History:")
            print("─" * 42)
            if not memory:
                print("  (empty)")
            for turn in memory:
                role = "You  " if turn["role"] == "user" else "Agent"
                ts   = turn.get("timestamp", "")[:19]
                print(f"  [{ts}] {role}: {turn['content'][:120]}{'...' if len(turn['content']) > 120 else ''}")
            print("─" * 42 + "\n")
            continue

        try:
            run_agent(query, memory)
        except Exception:
            traceback.print_exc()


# =====================================================
# 🚀 ENTRY POINT
# =====================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query  = " ".join(sys.argv[1:])
        memory = load_memory()
        run_agent(query, memory)
    else:
        chat_loop()
