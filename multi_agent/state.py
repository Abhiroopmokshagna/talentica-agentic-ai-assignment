from typing import Any, Dict, List

from pydantic import BaseModel, Field

class AgnetState(BaseModel):
    """Shared state passed between every node in langgraph wrokflow"""

    user_input: str
    tasks: List[str] = Field(default_factory = list)
    agnet_handoff_message: str = ""
    agent_b_response: str = ""
    weather_data: Dict[str, Any] = Field(default_factory = dict)
    final_summary: str = ""