import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "./app/chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def process_query(query_text: str):
    try:
        # Se genera la función de embedding
        embedding_function = get_embedding_function()

        # Inicializamos la base de datos Chroma
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Realiza la búsqueda en la base de datos
        results = db.similarity_search_with_score(query_text, k=5)

        # Formatea el contexto para pasarlo al modelo
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Generar la respuesta con el modelo
        model = OllamaLLM(model="llama3.2-vision:11b")
        response_text = model.invoke(prompt)

        # Obtiene las fuentes de los documentos relevantes
        sources = [doc.metadata.get("id", None) for doc, _score in results]

        # Formatea la respuesta
        formatted_response = f"Response: {response_text}\nSources: {sources}"

        return {"response_text": response_text, "sources": sources}

    except Exception as e:
        # Manejo de errores
        print(f"Error processing query: {e}")
        return {"error": str(e)}
