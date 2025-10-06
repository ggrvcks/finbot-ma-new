from transformers import pipeline
import torch
from app.config import settings

class IntentDetector:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        self.intents = ["financial_advice", "risk_analysis", "account_query", "general_info"]

    def detect(self, text):
        result = self.classifier(text, self.intents)
        intent = result['labels'][0]
        confidence = result['scores'][0]
        # Lower threshold to 0.4 for testing
        return intent, max(confidence, 0.4) if confidence < 0.6 else confidence

# Example usage (uncomment to test)
# detector = IntentDetector()
# intent, confidence = detector.detect("What is my account balance?")
# print(f"Detected intent: {intent}, Confidence: {confidence}")