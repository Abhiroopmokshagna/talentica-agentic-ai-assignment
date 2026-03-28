# Multi-Agent Weather Assistant

A two-agent AI system built with **LangGraph** and **Azure ChatOpenAI** that takes a
natural-language weather request, breaks it into tasks, fetches live data, and returns
a concise summary.

---

## Setup

### 1. Clone / enter the project directory

```bash
cd copilot-agentic-ai-tl
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure credentials

Copy the template and fill in your keys:

```bash
cp .env.template .env
```

Open `.env` and replace the placeholder values:

```env
AZURE_OPENAI_API_KEY=<your Azure OpenAI key>
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o          # name of your deployed model
AZURE_OPENAI_API_VERSION=2024-02-01
OPENWEATHER_API_KEY=<your OpenWeatherMap key>
```

---

## Running the system

```bash
python main.py
```

---

## Design & Agent Roles

### Architecture Overview

The system follows a **two-agent orchestration pattern** built on top of **LangGraph**. Execution flows as a directed acyclic graph (DAG) with four fixed nodes:

```
START → planner → executor → summarizer → END
```

All agents communicate through a single shared **`AgentState`** object that is passed between nodes. No agent calls another directly — coordination happens entirely through state mutations and the graph edges defined in `graph.py`.

---

### Shared State — `AgentState`

| Field                     | Type             | Purpose                                                         |
| ------------------------- | ---------------- | --------------------------------------------------------------- |
| `user_input`              | `str`            | The raw natural-language request from the user                  |
| `tasks`                   | `List[str]`      | Ordered task list produced by the Planner Agent                 |
| `agent_handoff_message`   | `str`            | Structured handoff note written by the Planner for the Executor |
| `executor_agent_response` | `str`            | Execution report written by the Executor for the Planner        |
| `weather_data`            | `Dict[str, Any]` | Raw weather results keyed by city name                          |
| `final_summary`           | `str`            | Human-readable summary produced by the Planner                  |

---

### Planner Agent (`PlannerAgent`)

**Responsibility:** Orchestration — breaking the user intent into tasks and compiling the final answer.

It operates across two graph nodes:

1. **`planner` node — `plan()`**

   - Receives the raw `user_input`.
   - Uses a structured LLM call (`with_structured_output(TaskList)`) to decompose the request into an ordered list of task strings (e.g. `["get_weather:New York", "summarize"]`).
   - Writes a human-readable handoff message to `agent_handoff_message` to brief the Executor Agent.

2. **`summarizer` node — `summarize()`**
   - Reads the `executor_agent_response` and `weather_data` written by the Executor Agent.
   - Calls the LLM to produce a friendly, natural-language summary that directly answers the user's original request.
   - Writes the result to `final_summary`.

---

### Executor Agent (`ExecutorAgent`)

**Responsibility:** Tool execution — resolving each task in the plan into concrete results.

It operates on the **`executor` node** via `execute()`:

- Iterates over the `tasks` list produced by the Planner Agent.
- For each task, uses a structured LLM call (`with_structured_output(ToolDecision)`) to determine which tool to invoke and with what arguments.
- Dispatches to the appropriate tool:
  - **`fetch_weather`** — calls the OpenWeatherMap API via `httpx` and stores the result in `weather_data`.
  - **`delegate_to_planner_agent`** — recognises that `summarize` is the Planner Agent's responsibility and skips execution, logging the delegation.
- Compiles an execution report and writes it to `executor_agent_response`.

---

### Tool — `fetch_weather`

An async function that calls the **OpenWeatherMap REST API** (`/data/2.5/weather`) using `httpx`. It handles `401 Unauthorized`, `404 Not Found`, request timeouts, and unexpected errors gracefully, always returning a dict with either weather data or an `"error"` key.

---

### Data Flow Summary

```
User input
    │
    ▼
PlannerAgent.plan()      ─── produces tasks + handoff message
    │
    ▼
ExecutorAgent.execute()  ─── runs fetch_weather per city, writes weather_data + response
    │
    ▼
PlannerAgent.summarize() ─── reads weather_data, produces final_summary
    │
    ▼
Printed to terminal
```
---
### Sample Inputs & Outputs
```
Input: What's the current weather in delhi and provide a short summary of it.

Output: The current weather in Delhi, India, is quite pleasant. The temperature is 28.05°C, but it feels slightly cooler at 27.85°C. The sky is partly covered with broken clouds, and the humidity is at 42%, making it relatively comfortable. There's a gentle breeze with a wind speed of 2.06 meters per second. Enjoy your day in Delhi!
```
```
Input: What's the current weather in delhi and new york and provide a short summary.

Output: Here's the current weather summary for the cities you asked about:

**Delhi, India**: The temperature is 28.05°C with a "feels like" temperature of 27.85°C. The weather is characterized by broken clouds, with a humidity level of 42% and a gentle wind blowing at 2.06 meters per second.

**New York, USA**: It's cooler with a temperature of 4.16°C, but it feels like -0.28°C due to the wind. The sky has a few clouds, and the humidity is at 47%. The wind is stronger here, blowing at 6.26 meters per second.
```