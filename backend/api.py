from services.llm_service import LLMService
from services.query_builder_service import QueryBuilderService
from services.memory_service import MemoryService
from services.faiss_service import FAISSService

from flask import Flask, request, jsonify, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

llm_service = LLMService()
memory_service = MemoryService()
query_builder_service = QueryBuilderService(memory_service, llm_service)
faiss_service = FAISSService()

@app.post("/chat")
def chat():
    data = request.get_json() or {}

    query = (data.get("message") or "").strip()
    if not query:
        abort(400, "Missing 'message'")

    print(f"[QUERY]: {query}")

    rag_query = query_builder_service.build_rag_query(query)
    print(f"[RAG QUERY]: {rag_query}")

    context = faiss_service.get_context(rag_query)

    prompt = query_builder_service.build_prompt(context, rag_query)
    print(f"[PROMPT]: {prompt}")

    try:
        answer = llm_service.post(prompt)
        print(f"[ANSWER]: {answer}")

        memory_service.add_memory(rag_query, context, answer)
        return jsonify({"reply": answer or "No response from model."})
    except RuntimeError as e:
        abort(503, str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
