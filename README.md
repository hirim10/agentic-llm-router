# agentic-llm-router
A CLI-based agentic AI system where **Groq acts as both the brain and one of the executors**. It intelligently routes each query — or parts of it — to either Groq or OpenRouter depending on complexity and token count.

---

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                🧠 GROQ BRAIN (llama-3.1-8b-instant)             │
│                                                                 │
│   STEP 1 — DECOMPOSE                                            │
│   Breaks query into 1–4 focused sub-questions                   │
│   (uses last 6 turns from memory.json as context)               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
   [sub-question 1] [sub-question 2] [sub-question 3]
          │               │               │
          ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                🧠 GROQ BRAIN — STEP 2: ROUTE                    │
│                                                                 │
│   For each sub-question, brain scores:                          │
│   • complexity_score  (1–10)                                    │
│   • estimated_tokens  (int)                                     │
│                                                                 │
│   complexity ≤ 6  AND  tokens ≤ 800  →  GROQ                   │
│   complexity > 6  OR   tokens > 800  →  OPENROUTER              │
└────────────┬────────────────────────────┬───────────────────────┘
             │                            │
             ▼                            ▼
┌────────────────────────┐   ┌────────────────────────────────────┐
│   ⚡ GROQ EXECUTOR     │   │   🌐 OPENROUTER EXECUTOR           │
│                        │   │                                    │
│  • llama-3.1-8b-instant│   │  Brain picks best model:           │
│    (simple queries)    │   │  • deepseek/deepseek-r1            │
│                        │   │    → Math, logic, reasoning        │
│  • llama-3.3-70b       │   │  • anthropic/claude-3.5-sonnet     │
│    (moderate queries)  │   │    → Code, technical writing       │
│                        │   │  • openai/gpt-4o                   │
│  💳 Also used as       │   │    → General complex queries       │
│  fallback if           │   │  • google/gemini-pro-1.5           │
│  OpenRouter has        │   │    → Long context / documents      │
│  no credits (402)      │   │                                    │
└────────────┬───────────┘   └──────────────┬─────────────────────┘
             │                              │
             └──────────────┬───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              🧠 GROQ BRAIN — STEP 3: SELF-REFLECT               │
│                                                                 │
│   Brain scores the answer (1–10)                                │
│                                                                 │
│   score ≥ 7  →  ✅ Accepted                                     │
│   score < 7  →  ⚠️  Retry with stronger model (up to 3x)       │
│                                                                 │
│   Escalation order:                                             │
│   gpt-4o → claude-3.5-sonnet → deepseek-r1 → gemini-pro-1.5   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              🧠 GROQ BRAIN — STEP 4: SYNTHESIZE                 │
│                                                                 │
│   Merges all sub-answers into one clean, unified final answer   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     💾 MEMORY (memory.json)                     │
│                                                                 │
│   Saves every user query + agent answer with timestamp          │
│   Loaded automatically on next run for context                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Routing Logic Summary

| Condition | Executor | Model |
|---|---|---|
| complexity ≤ 6 AND tokens ≤ 800 | ⚡ Groq | `llama-3.1-8b-instant` (simple) or `llama-3.3-70b-versatile` (moderate) |
| complexity > 6 OR tokens > 800 | 🌐 OpenRouter | Brain picks best model |
| OpenRouter returns 402 (no credits) | ⚡ Groq fallback | `llama-3.3-70b-versatile` |

---

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create your `.env` file
```bash
cp .env.example .env
```
Then fill in your keys:
```
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## ▶️ Running

### Interactive chat loop
```bash
python main.py
```

### Single question (one-shot mode)
```bash
python main.py "your question here"
```

### Inside the chat loop
| Command | What it does |
|---|---|
| Type your question | Agent processes it |
| `memory` | View full conversation history |
| `exit` or `quit` | Exit the agent |

---

## 📁 Project Structure

```
.
├── main.py          # Full agent code
├── memory.json      # Auto-created, stores conversation history
├── .env             # Your API keys (never commit this)
├── .env.example     # Template for .env
└── requirements.txt # Python dependencies
```

---

## 🔑 API Keys

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com |
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys |

---

## 🧩 What Each Part Does

**Brain (`llama-3.1-8b-instant`)** — runs 4 times per query:
1. Decomposes the query into sub-questions
2. Routes each sub-question to the right executor + model
3. Reflects on each answer and decides if it's good enough
4. Synthesizes all answers into one final response

**Groq Executor** — fast and free, handles simple to moderate queries

**OpenRouter Executor** — handles complex, long, or specialized queries using the best model for the job

**Memory** — persists all conversations to `memory.json` so the brain has context across sessions
