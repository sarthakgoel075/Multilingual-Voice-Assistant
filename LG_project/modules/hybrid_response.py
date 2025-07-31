import json
import sys
sys.path.append("/content/drive/MyDrive/LG_project")

from langchain.chains import RetrievalQA
from modules.sqlite_query import query_sqlite_db
from modules.mistral_loader import load_mistral
from modules.local_mistral_llm import LocalMistralLLM
from modules.vector_search import load_vector_index

mistral_path = "/content/drive/MyDrive/LG_project/mistral_local/mistral_local"
index_path = "/content/drive/MyDrive/LG_project/pdf_index"

model, tokenizer = load_mistral(mistral_path)
llm = LocalMistralLLM(model=model, tokenizer=tokenizer)

vectorstore = load_vector_index(index_path)
retriever = vectorstore.as_retriever()

def hybrid_answer(user_question: str) -> str:
    db_result = query_sqlite_db(user_question)
    docs = retriever.get_relevant_documents(user_question)
    doc_texts = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
You are an assistant that answers customer queries using ONLY the provided information.

---

USER QUESTION:
{user_question}

---

DATABASE INFO:
{db_result or "No relevant data."}

---

DOCUMENT INFO:
{doc_texts or "No relevant documents."}

---

INSTRUCTION:
Using ONLY the above information, respond in this *exact* JSON format:

{{
  "answer": "Your helpful answer in paragraph"
}}

⚠️ Do NOT include lists, nested structures, headings, or keys other than 'answer'.
⚠️ Do NOT include any explanations, markdown, or extra text outside this JSON block.

Now answer:
"""


    raw_response = llm(prompt)

    try:
        parsed = json.loads(raw_response)
        return parsed["answer"]
    except Exception as e:
        return f"❌ JSON parse error: {e}\nRaw output: {raw_response}"