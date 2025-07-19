from typing import List
from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """LLM verification of response quality using structured output"""

    reflects_context: bool = Field(
        description="Does response accurately reflect the context?"
    )
    addresses_query: bool = Field(description="Does response address the user's query?")
    is_clarification_question: bool = Field(
        description="Is the response asking for clarification from the user?"
    )
    confidence: float = Field(description="Overall confidence score (0-1)")
    issues: List[str] = Field(description="Any identified issues or concerns")
    rag_score: float = Field(description="RAG confidence score based on context relevance (0-1)")
    consistency_score: float = Field(description="Consistency score based on response-context alignment (0-1)")
    should_retry: bool = Field(description="Should the response be retried?")
    retry_reason: str = Field(description="Reason for retry if should_retry is True")