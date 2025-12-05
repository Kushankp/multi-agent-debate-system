# 🧠 Multi-Agent Debate System

## 📘 Overview
The **Multi-Agent Debate System** explores how multiple AI agents engage in structured debates to improve reasoning quality, factual grounding, and alignment.  
Two debating agents — **Agent A (Wikipedia RAG)** and **Agent B (Reddit RAG)** — generate arguments from different retrieval domains, while a **Judge Agent** evaluates their responses using scoring metrics and live web verification.

This system investigates how retrieval-source diversity affects factual accuracy, coherence, and argument strength in LLM debates.

---

## ⚙️ Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI** — debate orchestration API  
- **SentenceTransformers** — embedding generation  
- **Pinecone** — vector database for retrieval  
- **Groq / OpenAI API** — debate and judging logic  

### Frontend
- **React** — real-time debate visualization interface  
- **Axios** — API communication  

---

## 🚀 Key Features
- **Two-Agent RAG Debate:** Wikipedia-grounded vs. Reddit-grounded retrieval  
- **Judge Agent:** Scores factuality, coherence, and argument strength (0–5)  
- **Structured Debate Protocol:** Multi-round argument flow  
- **Retrieval-Augmented Reasoning:** Domain-specific evidence for improved grounding  

---

## 📊 Evaluation Summary

A total of **17 debates** were conducted:

- **8 Opinion/Sentiment-Based**
- **9 Factual/Technical-Based**

### 🟦 Overall Results
| Metric | Value |
|--------|-------|
| Total Debates | 17 |
| Successful Debates | 17 (100%) |
| Agent A (Wikipedia) Wins | 9 |
| Agent B (Reddit) Wins | 7 |
| Ties | 1 |

---

## 🟩 Opinion / Sentiment-Based Debates  
Agent B (Reddit) excels in subjective, community-driven discussions.

| Metric | Agent A (Wiki) | Agent B (Reddit) |
|--------|----------------|------------------|
| Wins | 1 | 7 |
| Avg Factuality | 2.4 | 3.1 |
| Avg Coherence | 2.5 | 3.2 |
| Avg Argument Strength | 2.4 | 3.0 |

**Finding:**  
Reddit’s diverse perspectives provide stronger support for opinion-based topics.

---

## 🟦 Factual / Technical Debates  
Agent A (Wikipedia) dominates structured, knowledge-heavy topics.

| Metric | Agent A (Wiki) | Agent B (Reddit) |
|--------|----------------|------------------|
| Wins | 9 | 0 |
| Avg Factuality | 3.2 | 2.1 |
| Avg Coherence | 3.1 | 2.2 |
| Avg Argument Strength | 3.0 | 2.1 |

**Finding:**  
Wikipedia’s verified information creates stronger factual grounding and coherence.

---

## 🧪 Judge Score Averages (All Debates)
| Agent | Factuality | Coherence | Argument Strength |
|-------|------------|-----------|-------------------|
| Agent A (Wikipedia) | 2.85 | 2.82 | 2.78 |
| Agent B (Reddit) | 2.65 | 2.75 | 2.68 |

---

## 🧠 Research Insight
- Retrieval **domain matters** — factual tasks favor structured sources, while subjective tasks favor informal/social sources.  
- Domain-diverse retrieval leads to **higher evidence coverage** and more balanced debates.  
- Heterogeneous RAG agents help expose reasoning gaps and retrieval bias.

---

## 🔮 Future Extensions
- Hybrid retrieval combining Reddit + Wikipedia  
- Automatic query-based source selection  
- Fine-tuning agents on domain-specific corpora  
- Expanded sources (academic papers, news, policy documents)  
- Human evaluation for debate quality assessment  

