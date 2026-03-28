from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import ExecutorAgent, PlannerAgent
from .state import AgentState

def build_graph() -> CompiledStateGraph:

    planner_agent = PlannerAgent()
    executor_agent = ExecutorAgent()

    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("planner", planner_agent.plan)
    graph.add_node("executor", executor_agent.execute)
    graph.add_node("summarizer", planner_agent.summarize)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()