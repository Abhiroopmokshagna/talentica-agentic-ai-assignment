import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

from .state import AgentState
from .tools import fetch_weather


def _create_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0,
    )


class TaskList(BaseModel):
    tasks: List[str]


class ToolDecision(BaseModel):
    tool: str
    city: Optional[str] = None


class PlannerAgent:

    def __init__(self) -> None:
        self.llm: AzureChatOpenAI = _create_llm()

    async def plan(self, state: AgentState) -> Dict[str, Any]:

        system_prompt = (
            "You are a task-planning agent. Analyse the user request and produce "
            "an ordered JSON array of task strings that fully satisfy the request. \n\n"
            "Available task types:\n"
            "get_weather:<city_name> - fetches current weather for city_name\n"
            "summarize - compile the collected data into a friendly summary\n\n"
            "Rules:\n"
            " 1. Always place 'summarize' as the very last element. \n"
            " 2. Extract the exact city name written in the request. \n"
            " 3. Return ONLY a valid JSON array - NO markdown. \n\n"
            ' Example: ["get_weather:New York", "summarize"]'
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.user_input)
        ]

        structured_llm = self.llm.with_structured_output(TaskList)
        result: TaskList = await structured_llm.ainvoke(messages)
        tasks: List[str] = result.tasks

        handoff_message = (
            "Planner Agent to Executor Agent delegation. \n"
            f"Original request: {state.user_input} \n"
            "Please execute the following tasks and return all results: \n"
            + "\n".join(f"{t}" for t in tasks)
        )

        print(f"PlannerAgent: taks created: {tasks}")
        print(f"PlannerAgent: Handoff message: {handoff_message}")
        return {"tasks": tasks, "agent_handoff_message": handoff_message}

    async def summarize(self, state: AgentState) -> Dict[str, Any]:
        agent_b_response = state.agent_b_response or "No response from Executor Agent"
        print(
            f"PlannerAgent: Reading Executor Agent response: {agent_b_response}")

        weather_json = json.dumps(state.weather_data, indent=2)
        user_context = (
            f"Original user request: \n{state.user_input}\n\n"
            f"Executor Agent execution report: \n{agent_b_response}\n\n"
            f"Weather data from Executor Agent: \n{weather_json}"
        )

        system_prompt = (
            "You are a helpful assistant. Using the weather data provided, "
            "write a friendly summary that directly answers the user's request. "
            "If any city entry contains 'error' field, acknowledge it clearly "
            "and suggest the user to verify the city name or API key."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_context)
        ]

        response = await self.llm.ainvoke(messages)
        print("PlannerAgent: Final summary generated")
        return {"final_summary": response.content}


