"""API-key-free LangChain orchestration for local MOSAIC-CRS inference.

LangChain keeps the MOSAIC recommender as a StructuredTool and optionally uses a
local Ollama chat model to turn the grounded tool result into a natural response.
If Ollama is unavailable, a deterministic response is returned so the application
still works without any paid service.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .mosaic_adapter import MosaicRecommendationAdapter


class RecommendationInput(BaseModel):
    query: str = Field(description="The user's current movie request or preference statement.")
    session_id: str = Field(description="Opaque browser conversation ID.")


_SYSTEM_PROMPT = """
You are the concise voice interface for MOSAIC-CRS, a ReDial movie recommender.
Use only the supplied MOSAIC tool result as factual grounding. Never invent titles,
genres, years, ratings, rankings, or reasons. Speak naturally in the user's language.
Keep the answer suitable for speech and normally under 90 words.

Rules:
- If policy_action is recommend, mention at most five supplied recommendations.
- If policy_action is ask_preference or clarify, ask one brief useful question.
- Do not read JSON keys, confidence scores, or implementation details aloud.
- If titles are placeholders such as "Movie 123", clearly say that the original
  ReDial item catalog is needed for readable titles.
- Never claim to be human.
""".strip()


class MosaicRealtimeToolChain:
    """Run MOSAIC locally and optionally verbalise its result with local Ollama."""

    def __init__(
        self,
        adapter: MosaicRecommendationAdapter,
        ollama_model: str = "llama3.2:3b",
        ollama_base_url: str = "http://127.0.0.1:11434",
        use_ollama: bool = True,
    ) -> None:
        self.adapter = adapter
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.use_ollama = bool(use_ollama)
        self._ollama_error: Optional[str] = None

        self.tool = StructuredTool.from_function(
            func=self._recommend,
            name="mosaic_recommend",
            description=(
                "Get grounded, personalised ReDial movie recommendations from "
                "the local MOSAIC-CRS checkpoint."
            ),
            args_schema=RecommendationInput,
        )
        self.tool_chain = RunnableLambda(self._normalise) | RunnableLambda(self.tool.invoke)
        self.llm_chain = None
        if self.use_ollama:
            try:
                from langchain_ollama import ChatOllama

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", _SYSTEM_PROMPT),
                        (
                            "human",
                            "Recent conversation:\n{history}\n\n"
                            "Current user message:\n{query}\n\n"
                            "Grounded MOSAIC result:\n{tool_result}\n\n"
                            "Give the spoken reply only.",
                        ),
                    ]
                )
                llm = ChatOllama(
                    model=self.ollama_model,
                    base_url=self.ollama_base_url,
                    temperature=0.2,
                    num_predict=220,
                )
                self.llm_chain = prompt | llm
            except Exception as exc:  # package missing or incompatible
                self._ollama_error = str(exc)
                self.llm_chain = None

    @staticmethod
    def _normalise(payload: Dict[str, Any]) -> Dict[str, str]:
        query = str(payload.get("query", "")).strip()
        session_id = str(payload.get("session_id", "")).strip()
        if not query:
            raise ValueError("Recommendation query is empty.")
        if not session_id:
            raise ValueError("Session ID is empty.")
        return {"query": query, "session_id": session_id}

    def _recommend(self, query: str, session_id: str) -> Dict[str, Any]:
        return self.adapter.recommend(query=query, session_id=session_id)

    def invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        result = self.tool_chain.invoke({"query": query, "session_id": session_id})
        if not isinstance(result, dict):
            raise RuntimeError("MOSAIC LangChain tool returned an invalid payload.")
        return result

    @staticmethod
    def _history_text(history: Iterable[Dict[str, str]]) -> str:
        rows: List[str] = []
        for turn in list(history)[-8:]:
            role = str(turn.get("role", "user")).strip().title()
            content = str(turn.get("content", "")).strip()
            if content:
                rows.append(f"{role}: {content}")
        return "\n".join(rows) or "No earlier turns."

    @staticmethod
    def _fallback_response(result: Dict[str, Any]) -> str:
        action = str(result.get("policy_action", "recommend"))
        recommendations = result.get("recommendations") or []
        explanation = str(result.get("model_explanation", "")).strip()

        if action == "end":
            return "Glad I could help. You can start a new session whenever you want another movie."
        if action == "ask_preference":
            return "What movie genres do you enjoy, and is there anything you want me to avoid?"
        if action == "clarify":
            return "Could you name one movie or genre you like and one you do not want?"
        if not recommendations:
            return "I need a little more information. What kind of movie are you in the mood for?"

        titles = [str(item.get("title") or item.get("item_id")) for item in recommendations[:5]]
        if len(titles) == 1:
            lead = f"My recommendation is {titles[0]}."
        else:
            lead = "My recommendations are " + ", ".join(titles[:-1]) + f", and {titles[-1]}."
        if explanation:
            return f"{lead} {explanation}".strip()
        return lead

    async def astream_response(
        self,
        query: str,
        session_id: str,
        history: Iterable[Dict[str, str]],
    ) -> AsyncIterator[str]:
        """Yield response text chunks after invoking the local MOSAIC tool."""
        result = await asyncio.to_thread(self.invoke, query, session_id)

        if self.llm_chain is None:
            yield self._fallback_response(result)
            return

        payload = {
            "history": self._history_text(history),
            "query": query,
            "tool_result": json.dumps(result, ensure_ascii=False, indent=2),
        }
        emitted = False
        try:
            async for chunk in self.llm_chain.astream(payload):
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "".join(
                        str(part.get("text", "")) if isinstance(part, dict) else str(part)
                        for part in content
                    )
                text = str(content or "")
                if text:
                    emitted = True
                    yield text
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._ollama_error = str(exc)
            if not emitted:
                yield self._fallback_response(result)

    def ollama_available(self, timeout: float = 0.8) -> bool:
        if not self.use_ollama:
            return False
        try:
            with urlopen(f"{self.ollama_base_url}/api/tags", timeout=timeout) as response:
                return 200 <= int(response.status) < 300
        except (OSError, URLError, ValueError):
            return False

    def status(self) -> Dict[str, Any]:
        return {
            "provider": "Ollama" if self.use_ollama else "deterministic template",
            "model": self.ollama_model if self.use_ollama else None,
            "base_url": self.ollama_base_url if self.use_ollama else None,
            "reachable": self.ollama_available(),
            "last_error": self._ollama_error,
            "fallback_enabled": True,
            "api_key_required": False,
        }
