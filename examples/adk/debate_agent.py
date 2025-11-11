import sys
import asyncio
from dotenv import load_dotenv
import os
from datetime import datetime

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content

# --- Load Environment Variables ---
load_dotenv()

# --- 1. Define Agent Factory ---
def create_debate_agent(index: int) -> LlmAgent:
    return LlmAgent(
        model="gemini-2.5-pro",
        name=f"DebateAgent_{index}",
        instruction=f"""
        You are DebateAgent_{index}, a specialist with a unique perspective.
        Your task is to analyze the intersection of the following topics:
        {{topics}}

        Present your viewpoint clearly, citing 3-5 key arguments or insights.
        Be constructive and persuasive, anticipating counterpoints.
        """,
        description=f"Debates the intersection of topics from perspective {index}.",
        tools=[google_search],
        output_key=f"agent_{index}_view"
    )

# --- 2. Define Consensus Agent ---
ConsensusAgent = LlmAgent(
    model="gemini-2.5-pro",
    name="ConsensusAgent",
    instruction="""
    You are the ConsensusAgent. You will receive multiple viewpoints from debate agents.
    Your task is to synthesize these into a unified, balanced consensus report.

    The report must include:
    1. A title
    2. A summary of the debated intersection
    3. Key points of agreement
    4. Remaining disagreements (if any)
    5. Final consensus statement

    Here are the viewpoints:
    {agent_0_view}
    {agent_1_view}
    {agent_2_view}
    """,
    description="Synthesizes agent viewpoints into a consensus report."
)

# --- 3. Define Coordinator Agent ---
def build_debate_coordinator(agent_count: int) -> SequentialAgent:
    debate_agents = [create_debate_agent(i) for i in range(agent_count)]
    return SequentialAgent(
        name="DebateCoordinator",
        sub_agents=debate_agents + [ConsensusAgent],
        description="Coordinates multi-agent debate and consensus synthesis."
    )

# --- 4. Define Runner ---
async def main(topics: list[str], agent_count: int = 3):
    topic_str = ", ".join(topics)
    print(f"\n--- Starting Multi-Agent Debate on Topics: {topic_str} ---")

    session_service = InMemorySessionService()
    app_name = "agents"  # Must match the agent directory
    user_id = "user_123"
    session_id = "session_456"

    # ✅ Create session with explicit session_id
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

    coordinator = build_debate_coordinator(agent_count)
    runner = Runner(agent=coordinator, app_name=app_name, session_service=session_service)

    # ✅ Pass context variable 'topics' explicitly
    user_message = Content(
        parts=[{'text': topic_str}],
        role="user",
        context={"topics": topic_str}
    )

    llm_response = None
    async for response_event in runner.run_async(
        new_message=user_message,
        user_id=user_id,
        session_id=session_id
    ):
        llm_response = response_event

    # --- Save and Display Output ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c for c in topic_str if c.isalnum() or c in (' ')).rstrip().replace(' ', '_')[:30]
    filename = f"{safe_topic}_{timestamp}_consensus.txt"

    consensus_text = llm_response.content.parts[0].text

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Topics: {topic_str}\n")
            f.write("-" * 40 + "\n\n")
            f.write(consensus_text)

        print("\n--- Consensus Report Saved ---")
        print(f"File: **{filename}**")
        print("-" * 40 + "\n")
        print(consensus_text)

    except Exception as file_error:
        print(f"\nERROR: Could not write to file {filename}. Error: {file_error}")
        print("\n--- Consensus Report (Console Fallback) ---")
        print(consensus_text)

# --- 5. CLI Entry Point ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debate_agent.py <topic1> <topic2> ... [--agents N]")
        sys.exit(1)

    args = sys.argv[1:]
    if "--agents" in args:
        idx = args.index("--agents")
        try:
            agent_count = int(args[idx + 1])
            topics = args[:idx]
        except:
            print("Error: '--agents' must be followed by an integer.")
            sys.exit(1)
    else:
        agent_count = 3
        topics = args

    try:
        asyncio.run(main(topics, agent_count))
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)