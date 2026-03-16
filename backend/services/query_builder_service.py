class QueryBuilderService:
    def __init__(self, memory_service, llm_service):
        self.memory_service = memory_service
        self.llm_service = llm_service

    def build_rag_query(self, query):
        query = (query or "").strip()
        if not query:
            return ""
        
        latest_memory = self.memory_service.get_latest_memory()

        if not latest_memory or not latest_memory.answer:
            return query

        rewrite_prompt = f"""Given the previous querions and answer and the user's new question,
        write a new search query that captures what the user is asking now but include previous 
        information if that helps complete the semantic of the new query.
        Output only the search query, nothing else.

Previous question:
{latest_memory.query}

Previous answer:
{latest_memory.answer}

New question:
{query}

Search query:"""
        try:
            rag_query = self.llm_service.post(rewrite_prompt).strip()
            if not rag_query:
                return query
            return rag_query
        except RuntimeError:
            return query

    def build_prompt(self, context, query):
        return f"""
Use the following context to answer the question. If the context does not contain relevant information, say so.
Context:
{context or '(No relevant context found.)'}

Question: {query}

Answer:"""