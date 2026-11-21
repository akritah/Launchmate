import pytest
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.user import User
from app.services.langgraph_agent import langgraph_onboarding_service


@pytest.mark.asyncio
async def test_langgraph_mermaid_diagram():
    diagram = langgraph_onboarding_service.get_mermaid_diagram()
    assert diagram is not None
    assert "graph TD" in diagram or "flowchart" in diagram
    assert "guardrail" in diagram
    assert "router" in diagram
    assert "rag_retrieval" in diagram
    assert "rag_generation" in diagram
    assert "action_executor" in diagram
    assert "escalation" in diagram
    assert "conversation" in diagram


@pytest.mark.asyncio
async def test_langgraph_guardrail_sensitive_branch():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(User).where(User.email == "sarah.chen@microsoft.com"))
        user = res.scalar_one()

        sensitive_query = "I am experiencing severe harassment and discrimination from my manager."
        result = await langgraph_onboarding_service.execute_turn(
            db=db,
            user=user,
            message=sensitive_query,
            conversation_id="test-langgraph-sens"
        )

        assert result["is_sensitive"] is True
        assert result["category"] == "escalate"
        assert "guardrail" in result["current_turn_steps"]
        assert "escalation" in result["current_turn_steps"]
        # Router & RAG must be bypassed
        assert "router" not in result["current_turn_steps"]
        assert "rag_retrieval" not in result["current_turn_steps"]
        assert len(result["response_content"]) > 20


@pytest.mark.asyncio
async def test_langgraph_knowledge_query_branch():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(User).where(User.email == "sarah.chen@microsoft.com"))
        user = res.scalar_one()

        knowledge_query = "What is the 401(k) retirement match policy and vesting period?"
        result = await langgraph_onboarding_service.execute_turn(
            db=db,
            user=user,
            message=knowledge_query,
            conversation_id="test-langgraph-rag"
        )

        assert result["category"] == "knowledge_query"
        assert result["current_turn_steps"] == ["guardrail", "router", "rag_retrieval", "rag_generation"]
        assert result["citations"] is not None
        assert len(result["citations"]) > 0
        assert "50%" in result["response_content"] or "match" in result["response_content"].lower()


@pytest.mark.asyncio
async def test_langgraph_action_executor_branch():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(User).where(User.email == "sarah.chen@microsoft.com"))
        user = res.scalar_one()

        action_query = "Please mark my MFA setup task as completed"
        result = await langgraph_onboarding_service.execute_turn(
            db=db,
            user=user,
            message=action_query,
            conversation_id="test-langgraph-action"
        )

        assert result["category"] == "task_action"
        assert result["current_turn_steps"] == ["guardrail", "router", "action_executor"]
        assert result["action_result"] is not None
        assert "task" in result["response_content"].lower() or "completed" in result["response_content"].lower()


@pytest.mark.asyncio
async def test_langgraph_conversation_branch():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(User).where(User.email == "sarah.chen@microsoft.com"))
        user = res.scalar_one()

        convo_query = "Good morning!"
        result = await langgraph_onboarding_service.execute_turn(
            db=db,
            user=user,
            message=convo_query,
            conversation_id="test-langgraph-convo"
        )

        assert result["category"] == "conversation"
        assert result["current_turn_steps"] == ["guardrail", "router", "conversation"]
        assert "Sarah" in result["response_content"] or "LangGraph" in result["response_content"]


@pytest.mark.asyncio
async def test_langgraph_state_checkpoint_persistence():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(User).where(User.email == "sarah.chen@microsoft.com"))
        user = res.scalar_one()

        conv_id = "test-langgraph-persist-session"

        # Turn 1: Greeting
        res1 = await langgraph_onboarding_service.execute_turn(
            db=db,
            user=user,
            message="Hello assistant",
            conversation_id=conv_id
        )
        assert res1["current_turn_steps"] == ["guardrail", "router", "conversation"]

        # Turn 2: Follow-up Knowledge query in same session
        res2 = await langgraph_onboarding_service.execute_turn(
            db=db,
            user=user,
            message="What is the home office allowance?",
            conversation_id=conv_id
        )
        assert res2["current_turn_steps"] == ["guardrail", "router", "rag_retrieval", "rag_generation"]
        # Accumulated steps across turns should reflect both turns
        assert len(res2["accumulated_steps"]) > len(res2["current_turn_steps"])
        assert "conversation" in res2["accumulated_steps"]
        assert "rag_generation" in res2["accumulated_steps"]
