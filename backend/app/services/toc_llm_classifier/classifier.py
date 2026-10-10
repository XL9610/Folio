"""LLM-based TOC classifier.

Uses an OpenAI-compatible client (DeepSeek, OpenAI, etc.) to classify
TOC entries as front matter or study content.
"""

from __future__ import annotations

import json
import logging

from ..llm import LLMConfig
from ..llm.client import create_llm_client, LLMJSONResponse
from .models import TOCClassificationResult, TOCEntryForLLM
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Limit to first N TOC entries to keep prompt size manageable
MAX_TOC_ENTRIES_FOR_LLM = 120


class LLMTOCClassifier:
    """Classify TOC entries using an LLM.

    Usage::

        config = LLMConfig(api_key="sk-...")
        classifier = LLMTOCClassifier(config)
        entries = [TOCEntryForLLM(...), ...]
        result = classifier.classify(entries)

    """

    def __init__(self, config: LLMConfig | None = None):
        self._config = config or LLMConfig()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = create_llm_client(self._config)
        return self._client

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def classify(
        self,
        entries: list[TOCEntryForLLM],
    ) -> TOCClassificationResult:
        """Classify TOC entries and return the first content index.

        Only sends the first MAX_TOC_ENTRIES_FOR_LLM entries to limit cost.
        """
        if not entries:
            return TOCClassificationResult(first_content_index=0, confidence=1.0)

        # Limit entries to avoid huge prompts
        limited = entries[:MAX_TOC_ENTRIES_FOR_LLM]

        # Format entries as JSON
        entries_json = json.dumps(
            [e.to_dict() for e in limited],
            indent=2,
            ensure_ascii=False,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(toc_json=entries_json)

        try:
            client = self._ensure_client()
            response = client.complete_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.warning("LLM TOC classification failed: %s", exc)
            return TOCClassificationResult(
                first_content_index=0,
                confidence=0.0,
                reasoning_summary=f"LLM call failed: {exc}",
            )

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_response(self, response: LLMJSONResponse) -> TOCClassificationResult:
        """Parse the LLM JSON response into a TOCClassificationResult."""
        data = response.data

        first_idx = data.get("first_content_index", 0)
        if not isinstance(first_idx, int) or first_idx < 0:
            first_idx = 0

        confidence = data.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        confidence = max(0.0, min(1.0, float(confidence)))

        skip = data.get("skip_indices", [])
        if not isinstance(skip, list):
            skip = []
        skip = [int(i) for i in skip if isinstance(i, (int, float))]

        keep = data.get("keep_indices", [])
        if not isinstance(keep, list):
            keep = []
        keep = [int(i) for i in keep if isinstance(i, (int, float))]

        reasoning = data.get("reasoning_summary", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        return TOCClassificationResult(
            first_content_index=first_idx,
            skip_indices=skip,
            keep_indices=keep,
            confidence=confidence,
            reasoning_summary=reasoning,
        )
