from worker_agents import retriever_agent, writer_agent, verifier_agent, run_model
from typing import Any, Dict
import json
from memory import (save_evidence, save_message, 
get_last_user_message, get_cached_evidence_context, summarize_cached_evidence)


# Tool Schema that guides the Orchestrator on how and when to call a worker agent  to perform its dedicated task.
ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "name": "reuse_cached_evidence",
        "description": "Activate the cached evidence from the previous turn when it is still directly relevant.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"}
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "call_retriever_agent",
        "description": "Retrieve relevant evidence chunks across the uploaded PDFs. Use a query that fully resolves any needed context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "call_writer_agent",
        "description": "Write the report using the retrieved evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {"type": "string"},
                "document_evidence": {"type": "string"},
            },
            "required": ["user_query", "document_evidence"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "call_verifier_agent",
        "description": "Verify whether the report draft is grounded in the retrieved evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "written_draft": {"type": "string"},
                "document_evidence": {"type": "string"},
            },
            "required": ["written_draft", "document_evidence"],
            "additionalProperties": False,
        },
    },
]

ORCHESTRATOR_INSTRUCTIONS =  (
        """
        You coordinate three agents to analyze information retrieved from a set of documents
        to generate a useful report.
        You may use the last user query and cached-evidence summary to decide whether the
        cached evidence is still relevant.
        If the question is generic, unrelated, or not grounded in the uploaded documents,
        do not call any tools and reply exactly:
        I can only answer questions grounded in the uploaded documents
        Before writing, choose exactly one evidence acquisition step for this turn:
        either reuse_cached_evidence or call_retriever_agent.
        If the current query needs different evidence, fresh pages, or a different comparison axis,
        call the retriever agent.
        If you call the retriever agent, provide a self-contained query.
        If no document evidence is active yet, do not call the writer agent.
        Then write the report.
        Then verify the report.
        Return the verified report as the final answer.
        """
    )

def format_orchestrator_state(state: dict[str, str]) -> str:
    return (
        f"- last_user_query: {state['last_user_query'] or 'None'}\n"
        f"- cached_evidence_query: {state['cached_evidence_query'] or 'None'}\n"
        f"- has_document_evidence: {bool(state['document_evidence'])}\n"
        f"- has_written_draft: {bool(state['written_draft'])}\n"
        f"- has_verification: {bool(state['verification'])}\n\n"
        f"Cached evidence summary:\n{state['cached_evidence_summary']}\n\n"
        f"Current document evidence:\n{state['document_evidence'] or 'None'}"
    )



def orchestrator_agent(
    user_query: str,
    session_id: str = "default",
    verbose: bool = True,
) -> Dict[str, Any]:
    cached_evidence = get_cached_evidence_context(session_id)
    last_user_query = get_last_user_message(session_id)

    state = {
        "user_query": user_query,
        "last_user_query": last_user_query or "",
        "cached_evidence_query": cached_evidence["query"] if cached_evidence else "",
        "cached_evidence_summary": cached_evidence["summary"] if cached_evidence else "None",
        "document_evidence": "",
        "written_draft": "",
        "verification": "",
        "final_answer": "",
    }


    response = run_model(
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        input_data=(
            f"User query: {user_query}\n\n"
            f"Current state:\n{format_orchestrator_state(state)}"
        ),
        tools=ORCHESTRATOR_TOOLS,
    )

    for _ in range(8):
        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            if not state["document_evidence"]:
                state["final_answer"] = response.output_text or "I can only answer questions provided in the uploaded documents"
                save_message(session_id, "user", user_query)
                save_message(session_id, "assistant", state["final_answer"])
                return state
            state["final_answer"] = state["verification"] or state["written_draft"] or response.output_text
            save_message(session_id, "user", user_query)
            save_message(session_id, "assistant", state["final_answer"])
            return state

        tool_outputs = []

        for call in function_calls:
            name = call.name
            args = json.loads(call.arguments)

            if name == "reuse_cached_evidence":
                if cached_evidence:
                    if verbose:
                        print("[ORCHESTRATOR] Reusing cached evidence.")
                    state["document_evidence"] = cached_evidence["evidence_json"]
                    state["cached_evidence_query"] = cached_evidence["query"]
                    output = f"Reused cached evidence: {args['reason']}"
                else:
                    output = "No cached evidence is available to reuse."

            elif name == "call_retriever_agent":
                if verbose:
                    print("[Retriever Agent] Retrieving documents...")
                output = retriever_agent(args["query"].strip() or state["user_query"]).model_dump_json()
                state["document_evidence"] = output
                state["cached_evidence_query"] = state["user_query"]
                save_evidence(session_id, state["user_query"], output)
                cached_evidence = {
                    "query": state["user_query"],
                    "evidence_json": output,
                    "summary": summarize_cached_evidence(output),
                }
                state["cached_evidence_summary"] = cached_evidence["summary"]

            elif name == "call_writer_agent":
                if not state["document_evidence"]:
                    output = "Cannot write yet because no document evidence has been activated. First reuse cached evidence or retrieve fresh evidence."
                else:
                    if verbose:
                        print("[Writer Agent] Writing report draft...")
                    output = writer_agent(
                        user_query=state["user_query"],
                        document_evidence=state["document_evidence"],
                    )
                    state["written_draft"] = output

            elif name == "call_verifier_agent":
                if not state["written_draft"]:
                    output = "Cannot verify yet because no report draft exists."
                else:
                    if verbose:
                        print("[Verifier Agent] Verifying report...")
                    output = verifier_agent(
                        written_draft=state["written_draft"],
                        document_evidence=state["document_evidence"],
                    )
                    state["verification"] = output

            else:
                output = "Unknown tool name."

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

        response = run_model(
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            input_data=[
                *tool_outputs,
                {
                    "type": "message",
                    "role": "user",
                    "content": f"Updated state:\n{format_orchestrator_state(state)}",
                },
            ],
            tools=ORCHESTRATOR_TOOLS,
            previous_response_id=response.id,
        )

    if state["verification"]:
        state["final_answer"] = state["verification"]
        save_message(session_id, "user", user_query)
        save_message(session_id, "assistant", state["final_answer"])
    else:
        state["final_answer"] = "Stopped because the maximum number of orchestration steps was reached."

    if verbose:
        print("[ORCHESTRATOR] Workflow complete.")
    return state
