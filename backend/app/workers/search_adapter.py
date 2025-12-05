from tavily import TavilyClient
import os

client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

def web_search_tavily(query: str, top_k: int = 5):
    """
    Tavily Web Search Adapter for Judge Agent
    Returns list of {title, url, snippet, published}
    """
    results = client.search(query=query, max_results=top_k)
    formatted = []
    for r in results.get("results", []):
        formatted.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("content") or r.get("snippet") or "",
            "published": r.get("published_date") or "unknown"
        })
    return formatted