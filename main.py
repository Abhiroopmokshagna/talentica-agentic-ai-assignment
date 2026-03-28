import asyncio
import logging

from dotenv import load_dotenv
from multi_agent.graph import build_graph
from multi_agent.state import AgentState

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


async def main() -> None:
    load_dotenv()
    app = build_graph()

    user_input = input("User: ").strip()
    if not user_input:
        print("No input. Exiting.")
        return
    initial_state = AgentState(user_input=user_input)
    result: AgentState = await app.ainvoke(initial_state)

    cities = [t.split(":", 1)[1]
              for t in result["tasks"] if t.startswith("get_weather:")]
    separator = "=" * 60
    bullet_tasks = "\n".join(f"  • {t}" for t in result["tasks"])
    bullet_cities = "\n".join(f"  • {c}" for c in cities)

    print(f"\n{separator}")
    print("  Workflow Complete")
    print(separator)
    print(f"Tasks executed :\n{bullet_tasks}")
    print(f"Cities queried :\n{bullet_cities}")
    print("\n--- Final Summary ---")
    print(result["final_summary"])
    print(separator)


if __name__ == "__main__":
    asyncio.run(main())
