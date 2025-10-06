import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    LLM_MODEL = "llama3"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DATA_PATH = "data/datasets/"
    N8N_WEBHOOK = "http://localhost:5678/webhook"
    LANGUAGES = ["en", "fr", "ar"]

settings = Settings()