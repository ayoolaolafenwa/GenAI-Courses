from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from qdrant_vector_database import similarity_search_per_document
from clients import get_openai_client

MODEL_NAME = "gpt-5.2-chat-latest"

# handler to load model
def run_model(
        *,
        instructions: str,
        input_data: Any,
        tools: Optional[List[Dict[str, Any]]] = None,
        previous_response_id: Optional[str] = None,
        model: str = MODEL_NAME
):
    client = get_openai_client()
    return client.responses.create(
        model=model,
        instructions=instructions,
        input=input_data,
        tools=tools or [],
        tool_choice="auto",
        previous_response_id=previous_response_id,
        parallel_tool_calls=False,
    )

# tool to be called to retrieve relevant information for a given query
def document_retrieval_tool(
    query: str,
    per_doc_topk: int = 4,
    score_threshold: float | None = 0.2
) -> Dict[str, Any]:
    results = similarity_search_per_document(
        query=query,
        per_doc_topk=per_doc_topk,
        score_threshold=score_threshold
    )

    #results = sorted(results, key=lambda item: item["score"], reverse=True)[:final_topk]

    chunks = []
    for item in results:
        chunks.append(
            {
                "document_name": item["document_name"],
                "page_number": int(item["page_number"]),
                "chunk_id": item["chunk_id"],
                "citation": item["citation"],
                "content": item["content"],
                "score": float(item["score"]),
            }
        )

    return {
        "query": query,
        "chunks": chunks,
    }

# Schemas to present retrieved context in structured form
class DocumentChunk(BaseModel):
    document_name: str
    page_number: int
    chunk_id: str
    citation: str
    content: str
    score: float

class DocumentEvidencePack(BaseModel):
    query: str
    summary: str
    chunks: List[DocumentChunk]

""" 
Retriever Agent
=============================
Calls the document retrieval tool and retrieve relevant context for a given query
and return them in a structured form 
"""
def retriever_agent(query: str) -> DocumentEvidencePack:
    tool_result = document_retrieval_tool(query, score_threshold=0.4)

    chunks = [
        DocumentChunk(
            document_name=c["document_name"],
            page_number=c["page_number"],
            chunk_id=c["chunk_id"],
            citation=c["citation"],
            content=c["content"],
            score=c["score"],
        )
        for c in tool_result["chunks"]
    ]

    summary = (
        """ 
        Retrieved relevant evidence across the uploaded PDF set. 
        Use these chunks for content creation.
        """
    )

    return DocumentEvidencePack(
        query=query,
        summary=summary,
        chunks=chunks,
    )

"""
Writer Agent
================================================
Writes structured information based on the information
obtained from the retrieval agent.
"""
def writer_agent(user_query: str, document_evidence: str) -> str:
    instructions = (
        """
        You write a clear report using only the provided document evidence.
        Answer the user's question directly in the opening lines.
        When the user asks for a judgment, comparison, recommendation, or conclusion, state the best-supported answer clearly.
        If the user asks a short follow-up question, respond with a focused, brief answer that addresses only the needed point.
        Then explain the reasoning using only the requested criteria.
        Do not add facts that are not in the evidence.
        If the evidence is weak or incomplete, say that clearly.
        """
    )

    input_text = (
        f"User query: {user_query}\n\n"
        f"Document evidence:\n{document_evidence}"
    )

    response = run_model(
        instructions=instructions,
        input_data=input_text,
        tools=None,
    )
    return response.output_text


"""
Verifier Agent
================================================
Verifies the content and claims of the report generated
by the writer agent. 

"""
def verifier_agent(written_draft: str, document_evidence: str) -> str:
    instructions = (
        """
        You check the draft against the provided evidence and return the final report.
        Make sure the answer responds directly to the user's question.
        When the user asks for a judgment, comparison, recommendation, or conclusion, state the best-supported answer clearly in the opening lines if the evidence supports it.
        If the user asks a short follow-up question, return a focused, brief answer that addresses only the needed point.
        Keep only statements that are supported by the evidence.
        Add citations at the end of supported sentences using the exact citation strings from the evidence, for example [file.pdf p.2].
        If the evidence is not strong enough for a confident conclusion, say that clearly.
        Keep the writing concise, clear, and natural.
        """
    )

    input_text = (
        f"Report draft:\n{written_draft}\n\n"
        f"Document evidence:\n{document_evidence}"
    )

    response = run_model(
        instructions=instructions,
        input_data=input_text,
        tools=None,
    )
    return response.output_text
