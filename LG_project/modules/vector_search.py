from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vector_index(index_path: str):
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = FAISS.load_local(
        index_path,
        embedding,
        allow_dangerous_deserialization=True
    )
    return vectorstore
