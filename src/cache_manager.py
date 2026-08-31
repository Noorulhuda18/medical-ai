"""
cache_manager.py
-----------------
Demonstrates BOTH caching strategies required by the assignment:

- InMemoryCache : lives only in RAM. Fastest, but wiped when the app
  restarts. Good for "one Streamlit session, don't repeat myself".
- SQLiteCache   : lives in a small .db file on disk. Slightly slower than
  memory, but survives an app restart. Good for "cache across sessions /
  across app restarts" so repeated identical patient inputs stay fast even
  after redeploying.

LangChain's `set_llm_cache(...)` is a GLOBAL switch: once you call it,
every subsequent LLM call in the process checks that cache automatically
before hitting the API, using the exact prompt text as the cache key.
"""

from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

SQLITE_CACHE_PATH = ".langchain_cache.db"

# Keep track of what's currently active so the UI can display it.
_current_cache_name = "No caching"


def set_cache(choice: str) -> str:
    """
    Turn on the requested cache type, or turn caching off.

    choice: one of "No caching", "In-memory cache", "SQLite cache"
    returns: a short human-readable description of what is now active.
    """
    global _current_cache_name
    _current_cache_name = choice

    if choice == "In-memory cache":
        set_llm_cache(InMemoryCache())
        return "In-memory cache active (stored in RAM, cleared on restart)."

    if choice == "SQLite cache":
        set_llm_cache(SQLiteCache(database_path=SQLITE_CACHE_PATH))
        return f"SQLite cache active (stored at ./{SQLITE_CACHE_PATH}, survives restarts)."

    # "No caching" -> explicitly disable by setting cache to None.
    set_llm_cache(None)
    return "Caching disabled - every submission calls the API."


def current_cache_name() -> str:
    """Return the label of whichever cache is currently active."""
    return _current_cache_name


CACHE_EXPLANATION = """
**In-memory cache**: stored entirely in RAM. It is the fastest option, but the
cache is lost as soon as the Streamlit process restarts. Best when you just
want to avoid re-paying for the same request twice within one session (e.g.
the user double-clicks "Submit").

**SQLite cache**: stored in a `.db` file on disk. Slightly slower than
in-memory (a small disk read), but it persists across app restarts and even
across different users hitting the same server. Best when you want repeated,
identical requests to stay fast over the lifetime of the deployed app rather
than just one session.

Both caches key off the *exact* prompt text sent to the model, so submitting
the exact same form twice will be visibly faster the second time whenever
either cache is enabled.
"""
