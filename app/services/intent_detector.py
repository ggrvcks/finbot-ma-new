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
        self.intents = ["financial_advice", "risk_analysis", "account_query", "general_info", "economic_data"]

    def detect(self, text):
        result = self.classifier(text, self.intents)
        intent = result['labels'][0]
        confidence = result['scores'][0]
        # Boost confidence for economic terms
        if "inflation" in text.lower() or "rate" in text.lower():
            intent = "economic_data" if confidence > 0.3 else intent
            confidence = max(confidence, 0.4)
        return intent, confidence

# Example usage (uncomment to test)
# detector = IntentDetector()
# intent, confidence = detector.detect("What is the inflation rate in Morocco?")
# print(f"Detected intent: {intent}, Confidence: {confidence}")