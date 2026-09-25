from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from .rag import _context_and_sources, _llm


KPI_QUERY = """
Find the most important financial metrics in the uploaded financial documents,
including revenue, net profit, EBITDA, operating profit, EPS, total assets,
total debt, cash and equivalents, operating cash flow, and profit margin.

Prefer figures that are explicitly reported in the documents.
"""


KPI_PROMPT = """
You are a financial document extraction system.

Extract financial KPIs ONLY from the supplied document context.

Return ONLY valid JSON.

The JSON must have this structure:

kpis:
  - name: Revenue
    value: reported value
    period: reporting period
    source: source filename
    page: page number

Rules:

1. Never invent a value.
2. Only include a KPI if the document contains supporting information.
3. Preserve the original currency and units whenever possible.
4. Preserve the reporting period.
5. Do not calculate values unless the document explicitly provides the result.
6. If the same KPI appears multiple times, prefer the clearest
   consolidated/latest reported figure.
7. Include the source filename and page when available.
8. Return an empty kpis list if no reliable KPIs can be extracted.
9. Do not provide investment advice.
10. Do not add Markdown or JSON code fences.

Document context:
{context}
"""


def _clean_json_response(text: str) -> str:
    """
    Remove accidental Markdown fences and isolate the JSON object.
    """
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "Gemini did not return a valid JSON object."
        )

    return text[start:end + 1]


def extract_financial_kpis(vector_store: Any) -> dict[str, Any]:
    """
    Retrieve relevant financial chunks and extract structured KPIs.
    """

    docs = vector_store.similarity_search(
        KPI_QUERY,
        k=10,
    )

    if not docs:
        return {
            "kpis": [],
            "sources": [],
        }

    context, sources = _context_and_sources(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                KPI_PROMPT,
            ),
            (
                "human",
                "Extract the financial KPIs from the context and return only JSON.",
            ),
        ]
    )

    chain = prompt | _llm()

    response = chain.invoke(
        {
            "context": context,
        }
    )

    raw = _clean_json_response(
        response.content
    )

    try:
        data = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned invalid JSON while extracting financial KPIs."
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "KPI response must be a JSON object."
        )

    kpis = data.get(
        "kpis",
        [],
    )

    if not isinstance(kpis, list):
        raise ValueError(
            "The KPI response must contain a 'kpis' list."
        )

    cleaned_kpis = []

    for item in kpis:

        if not isinstance(item, dict):
            continue

        name = str(
            item.get("name", "")
        ).strip()

        value = str(
            item.get("value", "")
        ).strip()

        if not name or not value:
            continue

        cleaned_kpis.append(
            {
                "name": name,
                "value": value,
                "period": str(
                    item.get(
                        "period",
                        "Not specified",
                    )
                ),
                "source": str(
                    item.get(
                        "source",
                        "Not specified",
                    )
                ),
                "page": str(
                    item.get(
                        "page",
                        "Not specified",
                    )
                ),
            }
        )

    return {
        "kpis": cleaned_kpis,
        "sources": sources,
    }