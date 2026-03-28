from .tools import fetch_weather
from .state import AgentState
import json
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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

        logger.info("[PlannerAgent]  Tasks generated : %s", tasks)
        logger.info("[PlannerAgent]  Handoff message : %r", handoff_message)
        return {"tasks": tasks, "agent_handoff_message": handoff_message}

    async def summarize(self, state: AgentState) -> Dict[str, Any]:
        executor_agent_response = state.executor_agent_response or "No response from Executor Agent"

        logger.info("[PlannerAgent]  Reading Executor Agent response: %s",
                    executor_agent_response)

        weather_json = json.dumps(state.weather_data, indent=2)

        user_context = (
            f"Original user request: \n{state.user_input}\n\n"
            f"Executor agent execution report: \n{executor_agent_response}\n\n"
            f"Weather data from executor agent: \n{weather_json}"
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
        logger.info("[PlannerAgent]  Final summary generated.")
        return {"final_summary": response.content}


class ExecutorAgent:

    _prompt = (
        "You are a helpful agent. Given a task string, determine which "
        "tool to invoke and return a JSON object with the tool name and its arguments. \n\n"
        "Avaialble tools:\n"
        "fetch_weather - retrieves current weather for a city.\n"
        'Response format: {"tool": "fetch_weather", "city": "<CityName>"}\n'
        "delegate_to_planner_agent - use this when the task is 'summarize', "
        "since summarisation is Planner Agent's responsibility."
        'Response format: {"tool": "delegate_to_planner_agent"}\n\n'
        "Return ONLY a valid JSON object. No markdown."
    )

    def __init__(self) -> None:
        self.llm: AzureChatOpenAI = _create_llm()

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        handoff_lines = "\n".join(
            f"  {line}" for line in (state.agent_handoff_message or "No hand off message").splitlines()
        )
        logger.info("[ExecutorAgent] Received handoff:\n%s", handoff_lines)

        weather_data: Dict[str, Any] = {}

        structured_llm = self.llm.with_structured_output(ToolDecision)

        for task in state.tasks:
            messages = [
                SystemMessage(content=self._prompt),
                HumanMessage(content=task)
            ]
            decision: ToolDecision = await structured_llm.ainvoke(messages)

            if decision.tool == "fetch_weather":
                city = (decision.city or "").strip()
                if not city:
                    logger.warning(
                        "[ExecutorAgent] fetch_weather selected but no city provided. Skipping task: %s", task)
                    continue
                logger.info(
                    "[ExecutorAgent] LLM chose fetch_weather → city: '%s'", city)
                result = await fetch_weather(city)
                weather_data[city] = result
                status = "error" if "error" in result else "success"
                logger.info(
                    "[ExecutorAgent] Weather fetch '%s': %s", city, status)
            elif decision.tool == "delegate_to_planner_agent":
                logger.info(
                    "[ExecutorAgent] Task '%s' is Planner Agent's responsibility — delegating back.", task)
            else:
                logger.warning(
                    "[ExecutorAgent] LLM chose unknown tool '%s'. Skipping.", decision.tool)

        weather_results: List[str] = []

        for city, data in weather_data.items():
            if "error" in data:
                weather_results.append(f"{city}: FAILED - {data['error']}")
            else:
                weather_results.append(
                    f"{city}: OK - {data['temperature_celcius']} degrees centigrade, {data['description']}"
                )

        summarize_delegation = any(t == "summarize" for t in state.tasks)

        if summarize_delegation:
            weather_results.append(
                "summarize: delegated to planner agent for compling summary"
            )

        executor_agent_response = (
            "Executor to Planner Agent Results.\n"
            f"Executed {len(state.tasks)} tasks: \n"
            + ("\n".join(weather_results)
               if weather_results else " (no tasks executed)")
        )

        response_lines = "\n".join(
            f"  {line}" for line in executor_agent_response.splitlines()
        )
        logger.info(
            "[ExecutorAgent] Response to Planner Agent:\n%s", response_lines)
        return {"weather_data": weather_data, "executor_agent_response": executor_agent_response}



