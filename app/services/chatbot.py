from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from app.services.intent_detector import IntentDetector
from app.services.rag_service import load_index
from app.config import settings

class Chatbot:
    def __init__(self):
        self.index = load_index()
        self.detector = IntentDetector()
        self.llm = None  # Will use Ollama's llama3 via CLI later

    def get_response(self, query):
        intent, confidence = self.detector.detect(query)
        if confidence < 0.6:  # Threshold for intent confidence
            return "Sorry, I didn’t understand. Could you rephrase?"

        if intent == "account_query":
            return "Account query detected. Please provide your account ID for balance details."
        elif intent == "financial_advice":
            response = self.index.as_query_engine().query("Provide financial advice based on economic data.")
            return str(response)
        elif intent == "risk_analysis":
            response = self.index.as_query_engine().query("Analyze economic risks based on recent data.")
            return str(response)
        elif intent == "general_info":
            response = self.index.as_query_engine().query(query)
            return str(response)
        return "I can help with financial advice, risk analysis, or account queries. Ask me anything!"

# Example usage (uncomment to test)
# chatbot = Chatbot()
# print(chatbot.get_response("What is the inflation rate in Morocco?"))