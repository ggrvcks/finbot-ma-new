from transformers import pipeline
import torch
from app.config import settings

class IntentDetector:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
        )
        self.intents = ["financial_advice", "risk_analysis", "account_query", "general_info"]

    def detect(self, text):
        result = self.classifier(text, self.intents)
        return result['labels'][0], result['scores'][0]

# Example usage (uncomment to test)
# detector = IntentDetector()
# intent, confidence = detector.detect("What is my account balance?")
# print(f"Detected intent: {intent}, Confidence: {confidence}")