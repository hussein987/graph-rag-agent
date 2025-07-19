import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.agents.graph_agent import create_agent
from src.evaluation.verification import VerificationResult
from src.utils.parsers import extract_entities, normalize_entity


class TestZeroHallucinationAgent(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.agent = create_agent()

    def test_entity_extraction(self):
        """Test entity extraction from text."""
        text = "Helsinki is the capital of Finland with 650,000 people."
        entities = extract_entities(text)

        # Should extract proper nouns
        self.assertTrue(any("Helsinki" in e for e in entities))
        self.assertTrue(any("Finland" in e for e in entities))

    def test_entity_normalization(self):
        """Test entity name normalization."""
        self.assertEqual(normalize_entity("Helsinki City"), "helsinki_city")
        self.assertEqual(normalize_entity("  FINLAND  "), "finland")

    def test_verification_result_structure(self):
        """Test VerificationResult structure."""
        result = VerificationResult(
            reflects_context=True,
            addresses_query=True,
            is_clarification_question=False,
            confidence=0.8,
            issues=[],
            rag_score=0.7,
            consistency_score=0.9,
            should_retry=False,
            retry_reason="",
        )

        # Check all required fields are present
        self.assertTrue(result.reflects_context)
        self.assertTrue(result.addresses_query)
        self.assertFalse(result.is_clarification_question)
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.rag_score, 0.7)
        self.assertEqual(result.consistency_score, 0.9)
        self.assertFalse(result.should_retry)
        self.assertEqual(result.retry_reason, "")





if __name__ == "__main__":
    unittest.main(verbosity=2)
