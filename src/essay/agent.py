import operator
import os
import sqlite3
from typing import Annotated, List, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient

from essay import prompts

_ = load_dotenv(find_dotenv())
# Inmemory ssqlite3 database
conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)


# Create state object
class AgentState(TypedDict):
    task: str
    lnode: str
    outline: str
    draft: str
    critique: str
    content: List[str]
    queries: List[str]
    count: Annotated[int, operator.add]
    revisions: int
    max_revisions: int


# pydantic model for strutured output
class Queries(BaseModel):
    queries: List[str] = Field(
        description="list of search queries that are will gather relevant information."
    )


# Agent class that implements the essay writing agent
class Agent:
    def __init__(
        self,
    ) -> None:
        # Model to use
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)

        # Prompts for nodes
        self.PLAN_PROMPT = prompts.OUTLINE_PROMPT
        self.WRITER_PROMPT = prompts.WRITER_PROMPT
        self.RESEARCH_PLAN_PROMPT = prompts.RESEARCH_PLAN_PROMPT
        self.REFLECTION_PROMPT = prompts.REFLECTION_PROMPT
        self.RESEARCH_CRITIQUE_PROMPT = prompts.RESEARCH_CRITIQUE_PROMPT

        # Search tool
        self.tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        # Build graph
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)
        builder.set_entry_point("planner")
        builder.add_conditional_edges(
            "generate", self.should_continue, {END: END, "reflect": "reflect"}
        )
        # add edges
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_after=[
                "planner",
                "generate",
                "reflect",
                "research_plan",
                "research_critique",
            ],
        )

    def plan_node(self, state: AgentState) -> dict:
        """node to generate an outline for the essay.
        Args:
            state (AgentState): State of the agent.
        Returns:
            dict: A dictionary with the outline and next node and count.
        """

        messages = [
            SystemMessage(content=self.PLAN_PROMPT),
            HumanMessage(content=state["task"]),
        ]
        response = self.model.invoke(messages)
        return {
            "outline": response.content,
            "lnode": "planner",
            "count": 1,
        }

    def research_plan_node(self, state: AgentState) -> dict:
        """Node to genrate a research plan based on the outline of the plan node

        Args:
            state (AgentState): state of the agent
        Returns:
            dict: dictionary eith the content, next node and count and the \
                generated queries.
        """  # noqa: E501
        queries = self.model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=self.RESEARCH_PLAN_PROMPT),
                HumanMessage(content=state["task"]),
            ]
        )
        content = state["content"] or []  # add to content
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
        return {
            "content": content,
            "queries": queries.queries,
            "lnode": "research_plan",
            "count": 1,
        }

    def generation_node(self, state: AgentState) -> dict:
        """Node to genearte the draft of the essay based on the
            outline and content gathered from the reserch plan node.

        Args:
            state (AgentState): state of the agent

        Returns:
            dict: a dictionary that returns the draft, next node, count and number of revisions.
        """  # noqa: E501
        content = "\n\n".join(state["content"] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['outline']}"
        )
        messages = [
            SystemMessage(content=self.WRITER_PROMPT.format(content=content)),
            user_message,
        ]
        response = self.model.invoke(messages)
        return {
            "draft": response.content,
            "revisions": state.get("revisions", 1) + 1,
            "lnode": "generate",
            "count": 1,
        }

    def reflection_node(self, state: AgentState) -> dict:
        """Node that critiques the draft generated by the generation node.

        Args:
            state (AgentState): state of agent.

        Returns:
            dict: A dictionary with the critique, last node and count.
        """
        messages = [
            SystemMessage(content=self.REFLECTION_PROMPT),
            HumanMessage(content=state["draft"]),
        ]
        response = self.model.invoke(messages)
        return {
            "critique": response.content,
            "lnode": "reflect",
            "count": 1,
        }

    def research_critique_node(self, state: AgentState) -> dict:
        """A ndoe that uses the critique from the reflection node
        to gather more information to improve the draft.

        Args:
            state (AgentState): state of the agent.

        Returns:
            dict: dictionary with the content, last node and count.
        """
        queries = self.model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=self.RESEARCH_CRITIQUE_PROMPT),
                HumanMessage(content=state["critique"]),
            ]
        )
        content = state["content"] or []
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
        return {
            "content": content,
            "lnode": "research_critique",
            "count": 1,
        }

    def should_continue(self, state: AgentState) -> str:
        if state["revisions"] > state["max_revisions"]:
            return END
        return "reflect"
