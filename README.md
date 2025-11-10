# 🧠 Multi-Agent Debate System

## 💡 Concept

The **Multi-Agent Debate System** is designed to explore how two AI agents can engage in structured debates on complex AI-related topics. Each debate involves two opposing agents — **Agent A (Pro)** and **Agent B (Con)** — with a **Judge Agent** evaluating their arguments to determine the winner.

The goal is to study how debate-based reasoning between multiple AI systems can improve factual grounding, reasoning depth, and alignment.

---

## ⚙️ Tech Stack

**Backend**

* Python 3.10+
* FastAPI — backend API framework
* SentenceTransformers — embedding generation
* Pinecone — vector database for retrieval-augmented generation (RAG)
* Groq / OpenAI API — large language model interaction and debate logic

**Frontend**

* React — user interface for debate visualization
* Axios — handles API communication

---

## 🚀 Features

* Two debating agents that alternate responses across multiple rounds (A → B → A → B)
* Judge agent that evaluates and scores debates
* Context-based retrieval for grounded responses
* Structured prompts for consistency and reasoning clarity

---

## 🧩 Research Goal

To analyze how adversarial and collaborative reasoning among AI agents can:

* Improve factual consistency
* Enhance reasoning depth
* Support better decision-making in LLM systems
