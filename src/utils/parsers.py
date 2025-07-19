import re
from typing import List, Tuple, Dict, Any
from langchain_openai import OpenAI
from src.utils.config import config


def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract (subject, relation, object) triples from text using LLM.
    Simple implementation - can be enhanced with more sophisticated NLP.
    """
    llm = OpenAI(
        openai_api_key=config.openai_api_key,
        temperature=0,
        model_name=config.llm_model_name,
    )

    prompt = f"""
    Extract factual triples in the format (subject, relation, object) from the following text.
    Return only the triples, one per line, in the format: subject|relation|object
    
    Text: {text}
    
    Triples:
    """

    response = llm.invoke(prompt)
    triples = []

    for line in response.strip().split("\n"):
        if "|" in line:
            parts = line.split("|")
            if len(parts) == 3:
                subject, relation, obj = [p.strip() for p in parts]
                triples.append((subject, relation, obj))

    return triples


def extract_facts(text: str) -> List[str]:
    """Extract key facts from text for consistency checking."""
    # Simple sentence splitting - can be enhanced
    sentences = re.split(r"[.!?]+", text)
    facts = [s.strip() for s in sentences if len(s.strip()) > 10]
    return facts


def normalize_entity(entity: str) -> str:
    """Normalize entity names for consistent graph lookup."""
    return entity.lower().strip().replace(" ", "_")


def extract_entities(text: str) -> List[str]:
    """Extract named entities from text."""
    # Simple regex-based entity extraction
    # In production, use spaCy or similar NLP library
    entity_patterns = [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Proper nouns
        r"\b\d{1,2}[:/]\d{1,2}(?:[:/]\d{2,4})?\b",  # Dates/times
        r"\b\$?\d+(?:,\d{3})*(?:\.\d{2})?\b",  # Numbers/prices
    ]

    entities = []
    for pattern in entity_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)

    return list(set(entities))
