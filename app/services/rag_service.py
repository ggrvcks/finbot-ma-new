from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.config import settings

def build_index():
    embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)
    documents = SimpleDirectoryReader(settings.DATA_PATH).load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir="data/index")
    return index

def load_index():
    storage_context = StorageContext.from_defaults(persist_dir="data/index")
    return load_index_from_storage(storage_context)