import logging
from typing import Dict, Any, List, Optional, TypedDict, Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.models.user import User
from app.models.chat_message import ChatMessage
from app.services.agent_router import agent_router_service
from app.services.rag_service import rag_service
from app.services.task_service import task_service
from app.services.escalation_service import escalation_service
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("langgraph_agent")
logger.setLevel(logging.INFO)


class OnboardingState(TypedDict, total=False):
    """
    Typed state schema for the LangGraph onboarding multi-agent graph.
    Maintains user context, conversation memory, classification, tool execution,
    grounded RAG context, and diagnostic step tracking.
    """
    # Session & User identity
    user_id: Optional[str]
    user_name: Optional[str]
    user_role: Optional[str]
    user_department: Optional[str]
    user_location: Optional[str]
    organization_id: str
    conversation_id: str

    # Conversation Turn
    query: str
    messages: List[Dict[str, Any]]
    category: str  # "knowledge_query" | "task_action" | "escalate" | "conversation"
    confidence: float
    reasoning: Optional[str]

    # Guardrails & Sensitivity
    is_sensitive: bool
    sensitivity_reason: Optional[str]

    # Tool / Action Execution
    detected_action: Optional[str]
    action_parameters: Optional[Dict[str, Any]]
    action_result: Optional[Dict[str, Any]]

    # Grounded RAG Knowledge
    rag_chunks: Optional[List[Dict[str, Any]]]
    citations: Optional[List[Dict[str, Any]]]

    # Output & Diagnostics
    response_content: str
    escalation_ticket_id: Optional[str]
    current_turn_steps: List[str]
    accumulated_steps: List[str]


# ─── GRAPH NODES ───

async def guardrail_node(state: OnboardingState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Deterministic safety node: Scans input against high-priority sensitive topics
    (harassment, compensation, visa/immigration, whistleblower, wellbeing).
    Bypasses LLM entirely when a sensitive topic is detected.
    """
    query = state.get("query", "")
    is_sensitive, reason = agent_router_service.check_sensitivity(query)
    turn_steps = list(state.get("current_turn_steps", [])) + ["guardrail"]
    accum_steps = list(state.get("accumulated_steps", [])) + ["guardrail"]

    if is_sensitive:
        logger.info(f"[LangGraph:guardrail] Sensitive topic flagged: {reason}")
        return {
            "is_sensitive": True,
            "sensitivity_reason": reason,
            "category": "escalate",
            "confidence": 1.0,
            "reasoning": f"Deterministic safety guardrail triggered: {reason}",
            "current_turn_steps": turn_steps,
            "accumulated_steps": accum_steps,
        }

    return {
        "is_sensitive": False,
        "sensitivity_reason": None,
        "current_turn_steps": turn_steps,
        "accumulated_steps": accum_steps,
    }


async def router_node(state: OnboardingState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Classification node: Categorizes the query into knowledge_query, task_action,
    conversation (chitchat/greeting), or escalate (ambiguity or low confidence <0.70).
    """
    query = state.get("query", "")
    user_id = state.get("user_id") or "anonymous"
    turn_steps = list(state.get("current_turn_steps", [])) + ["router"]
    accum_steps = list(state.get("accumulated_steps", [])) + ["router"]

    decision = await agent_router_service.route(query, user_id)
    logger.info(f"[LangGraph:router] Routed to '{decision.category}' (confidence={decision.confidence})")

    return {
        "category": decision.category,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "detected_action": decision.detected_action,
        "action_parameters": decision.action_parameters or {},
        "current_turn_steps": turn_steps,
        "accumulated_steps": accum_steps,
    }


async def rag_retrieval_node(state: OnboardingState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    RAG Retrieval node: Queries the pgvector/SQLite document store for top-k chunks
    matching the employee's query within their organization.
    """
    turn_steps = list(state.get("current_turn_steps", [])) + ["rag_retrieval"]
    accum_steps = list(state.get("accumulated_steps", [])) + ["rag_retrieval"]

    query = state.get("query", "")
    org_id = state.get("organization_id", "launchmate")
    db = config.get("configurable", {}).get("db") if config else None

    chunks_data = []
    if db:
        chunks = await rag_service.retrieve_relevant_chunks(db, query, organization_id=org_id, top_k=3)
        chunks_data = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "document_title": doc.title if doc else "Handbook",
                "section_title": chunk.section_title,
                "content": chunk.content,
                "similarity_score": score,
            }
            for chunk, doc, score in chunks
        ]
        logger.info(f"[LangGraph:rag_retrieval] Retrieved {len(chunks_data)} relevant chunks")

    return {
        "rag_chunks": chunks_data,
        "current_turn_steps": turn_steps,
        "accumulated_steps": accum_steps,
    }


async def rag_generation_node(state: OnboardingState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    RAG Generation node: Synthesizes a grounded employee answer strictly constrained
    to retrieved chunks, extracting and formatting verified citations.
    """
    turn_steps = list(state.get("current_turn_steps", [])) + ["rag_generation"]
    accum_steps = list(state.get("accumulated_steps", [])) + ["rag_generation"]

    query = state.get("query", "")
    org_id = state.get("organization_id", "launchmate")
    db = config.get("configurable", {}).get("db") if config else None

    if db:
        orm_chunks = await rag_service.retrieve_relevant_chunks(db, query, organization_id=org_id, top_k=3)
        answer, citations = await rag_service.generate_grounded_answer(query, orm_chunks)
        citations_data = [c.model_dump() for c in citations]
    else:
        answer = "I searched the corporate handbook repository but no active database session was provided."
        citations_data = []

    logger.info(f"[LangGraph:rag_generation] Generated grounded answer with {len(citations_data)} citations")

    return {
        "response_content": answer,
        "citations": citations_data,
        "current_turn_steps": turn_steps,
        "accumulated_steps": accum_steps,
    }


async def action_executor_node(state: OnboardingState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Autonomous Tool Node: Executes function-calling actions on behalf of the user
    (complete_task, list_tasks, raise_it_ticket, book_orientation, check_progress).
    """
    turn_steps = list(state.get("current_turn_steps", [])) + ["action_executor"]
    accum_steps = list(state.get("accumulated_steps", [])) + ["action_executor"]

    db = config.get("configurable", {}).get("db") if config else None
    user = config.get("configurable", {}).get("user") if config else None
    action_name = state.get("detected_action")
    params = state.get("action_parameters", {})

    if db and user and action_name:
        action_result = await task_service.execute_agent_action(
            db=db,
            user=user,
            action_name=action_name,
            parameters=params
        )
        msg = action_result.get("message", f"Action '{action_name}' executed successfully.")
    else:
        action_result = {
            "status": "simulated",
            "action": action_name,
            "message": f"Tool action '{action_name}' executed in sandbox mode."
        }
        msg = action_result["message"]

    logger.info(f"[LangGraph:action_executor] Executed action: {action_name}")

    return {
        "action_result": action_result,
        "response_content": msg,
        "current_turn_steps": turn_steps,
        "accumulated_steps": accum_steps,
    }


async def escalation_node(state: OnboardingState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Human-in-the-Loop Escalation Node: Captures conversation context snapshot,
    creates an #ESC priority ticket in the HR operations queue, and returns an
    empathetic handoff message to the employee.
    """
    turn_steps = list(state.get("current_turn_steps", [])) + ["escalation"]
    accum_steps = list(state.get("accumulated_steps", [])) + ["escalation"]

    db = config.get("configurable", {}).get("db") if config else None
    user = config.get("configurable", {}).get("user") if config else None
    query = state.get("query", "")
    conv_id = state.get("conversation_id", "default")
    reason = state.get("reasoning", "Escalated for human HR specialist review.")
    sensitivity = state.get("sensitivity_reason")

    if db and user:
        esc, reply = await escalation_service.create_escalation(
            db=db,
            user=user,
            conversation_id=conv_id,
            reason=reason,
            trigger_message=query,
            sensitivity_reason=sensitivity
        )
        ticket_id = esc.id
        action_data = {
            "type": "escalation_created",
            "escalation_id": esc.id,
            "ticket_id": esc.id[:6].upper(),
            "priority": esc.priority,
            "reason": esc.reason,
            "message": f"Priority Human Escalation #{esc.id[:6].upper()} opened."
        }
    else:
        ticket_id = "ESC-DEMO"
        reply = (
            f"I have flagged this matter ({sensitivity or reason}) and escalated it to our confidential "
            f"People Operations team. A specialist will follow up with you privately."
        )
        action_data = {"type": "escalation_created", "ticket_id": ticket_id, "reason": reason}

    logger.info(f"[LangGraph:escalation] Escalation ticket created: {ticket_id}")

    return {
        "escalation_ticket_id": ticket_id,
        "action_result": action_data,
        "response_content": reply,
        "current_turn_steps": turn_steps,
        "accumulated_steps": accum_steps,
    }


async def conversation_node(state: OnboardingState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Conversational / Small-Talk Node: Handles greetings, gratitude, farewells,
    and cheerful onboarding encouragement without invoking heavy RAG or escalation.
    """
    turn_steps = list(state.get("current_turn_steps", [])) + ["conversation"]
    accum_steps = list(state.get("accumulated_steps", [])) + ["conversation"]

    sub_type = state.get("detected_action") or "greeting"
    user_name = state.get("user_name") or "there"
    first_name = user_name.split()[0] if user_name else "there"

    convo_replies = {
        "greeting": (
            f"👋 Hey {first_name}! Welcome — I'm your Launch Mate onboarding assistant, orchestrated via LangGraph. "
            f"I can help guide you through company policies, health benefits, IT setup, your checklist tasks, or connect you with HR.\n\n"
            f"What can I help you with today?"
        ),
        "thanks": (
            f"You're very welcome, {first_name}! 😊 Happy to help. "
            f"If you have any more questions about your onboarding checklist, policies, or need anything else — just ask!"
        ),
        "farewell": (
            f"Goodbye {first_name}! 👋 Best of luck with your onboarding journey. "
            f"I'm here 24/7 whenever you need help — just open the chat anytime. Have a great day!"
        ),
        "chitchat": (
            f"I'm doing great, thanks for asking! 😄 I'm your AI onboarding assistant — "
            f"always ready to help you navigate your new role at Launch Mate.\n\n"
            f"Feel free to ask me about company policies, benefits, IT setup, or your sprint tasks!"
        ),
    }
    reply = convo_replies.get(sub_type, convo_replies["greeting"])

    return {
        "response_content": reply,
        "current_turn_steps": turn_steps,
        "accumulated_steps": accum_steps,
    }


# ─── CONDITIONAL ROUTING FUNCTIONS ───

def route_after_guardrail(state: OnboardingState) -> Literal["escalation", "router"]:
    """Branch immediately to escalation if deterministic safety check triggers."""
    if state.get("is_sensitive"):
        return "escalation"
    return "router"


def route_after_router(state: OnboardingState) -> Literal["escalation", "action_executor", "conversation", "rag_retrieval"]:
    """Branch based on router classification intent."""
    category = state.get("category")
    if category == "escalate":
        return "escalation"
    elif category == "task_action":
        return "action_executor"
    elif category == "conversation":
        return "conversation"
    return "rag_retrieval"


# ─── GRAPH CONSTRUCTION & COMPILATION ───

def build_onboarding_graph() -> StateGraph:
    """Builds and wires the full LangGraph state machine for onboarding."""
    workflow = StateGraph(OnboardingState)

    # Register Nodes
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("router", router_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("rag_generation", rag_generation_node)
    workflow.add_node("action_executor", action_executor_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("conversation", conversation_node)

    # Register Edges & Conditional Branches
    workflow.add_edge(START, "guardrail")

    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "escalation": "escalation",
            "router": "router",
        }
    )

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "escalation": "escalation",
            "action_executor": "action_executor",
            "conversation": "conversation",
            "rag_retrieval": "rag_retrieval",
        }
    )

    # RAG pipeline linear transition
    workflow.add_edge("rag_retrieval", "rag_generation")

    # Terminal edges
    workflow.add_edge("rag_generation", END)
    workflow.add_edge("action_executor", END)
    workflow.add_edge("escalation", END)
    workflow.add_edge("conversation", END)

    return workflow


class LangGraphOnboardingService:
    """
    Singleton service exposing the compiled LangGraph agent, thread memory checkpoints,
    Mermaid workflow visualization, and execution endpoints.
    """
    def __init__(self):
        self.workflow = build_onboarding_graph()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def get_mermaid_diagram(self) -> str:
        """Exports the state graph as a Mermaid flowchart for UI rendering."""
        return self.app.get_graph().draw_mermaid()

    async def execute_turn(
        self,
        db: AsyncSession,
        user: User,
        message: str,
        conversation_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Executes a single conversational turn through the compiled LangGraph state machine.
        Persists state checkpoint using thread_id=conversation_id.
        """
        initial_turn_state: OnboardingState = {
            "query": message,
            "user_id": user.id,
            "user_name": user.full_name or "Employee",
            "user_role": user.role or "Software Engineer",
            "user_department": user.department or "Engineering",
            "user_location": user.location or "Redmond, WA",
            "organization_id": getattr(user, "organization_id", "launchmate") or "launchmate",
            "conversation_id": conversation_id,
            "current_turn_steps": [],
        }

        config: RunnableConfig = {
            "configurable": {
                "thread_id": f"{user.id}:{conversation_id}",
                "db": db,
                "user": user,
            }
        }

        final_state = await self.app.ainvoke(initial_turn_state, config=config)

        # Update message history in state
        history = list(final_state.get("messages", []))
        history.append({"sender": "user", "content": message})
        history.append({"sender": "assistant", "content": final_state.get("response_content", "")})
        final_state["messages"] = history

        return final_state


# Singleton instance
langgraph_onboarding_service = LangGraphOnboardingService()
