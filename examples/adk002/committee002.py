import sys
import asyncio
import os
import glob
from datetime import datetime
from dotenv import load_dotenv

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content

# --- Load Environment Variables ---
load_dotenv()

# --- Load Previous Reports ---
def load_previous_reports(topic_name: str) -> str:
    reports = sorted(glob.glob(f"{topic_name}.report*.txt"))
    combined = ""
    for fname in reports:
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                combined += f"\n--- Report: {fname} ---\n"
                combined += f.read() + "\n"
        except Exception as e:
            print(f"⚠️ Could not read {fname}: {e}")
    return combined.strip()

# --- Save New Report with Incremental Filename ---
def save_committee_report(topic_name: str, content: str) -> str:
    existing = sorted(glob.glob(f"{topic_name}.report*.txt"))
    next_index = len(existing) + 1
    filename = f"{topic_name}.report{next_index:03d}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ Committee Report Saved: {filename}")
    except Exception as e:
        print(f"\n❌ ERROR saving report: {e}")
    return filename

# --- Define Debate Agent ---
def create_debate_agent(index: int, topic_str: str) -> LlmAgent:
    return LlmAgent(
        model="gemini-2.5-pro",
        name=f"DebateAgent_{index}",
        instruction=f"""
You are DebateAgent_{index}, a domain expert with a unique perspective.
Your task is to analyze the following management challenge:
{topic_str}

Present your viewpoint clearly, citing 3–5 key arguments or insights.
Be constructive and persuasive, anticipating counterpoints.
""",
        description=f"Debates the topic from perspective {index}.",
        tools=[google_search],
        output_key=f"agent_{index}_view"
    )

# --- Define Consensus Agent ---
def create_consensus_agent(agent_count: int, prior_reports: str, new_inputs: str) -> LlmAgent:
    views = "\n".join([f"{{agent_{i}_view}}" for i in range(agent_count)])
    return LlmAgent(
        model="gemini-2.5-pro",
        name="ConsensusAgent",
        instruction=f"""
You are the ConsensusAgent. Your task is to synthesize a new committee report.

You will receive:
1. Prior committee reports (historical context)
2. New management inputs (current concerns)
3. Viewpoints from expert agents (debate outputs)

Your report must include:
- Title
- Summary of current challenge and historical context
- Key agreements and disagreements
- Updated recommendations
- Notable changes from previous reports

--- Prior Reports ---
{prior_reports or 'None'}

--- New Management Inputs ---
{new_inputs}

--- Debate Agent Views ---
{views}
""",
        description="Synthesizes expert views and historical context into an updated committee report."
    )

# --- Define Coordinator Agent ---
def build_debate_coordinator(agent_count: int, topic_str: str, prior_reports: str) -> SequentialAgent:
    debate_agents = [create_debate_agent(i, topic_str) for i in range(agent_count)]
    consensus_agent = create_consensus_agent(agent_count, prior_reports, topic_str)
    return SequentialAgent(
        name="DebateCoordinator",
        sub_agents=debate_agents + [consensus_agent],
        description="Coordinates expert debate and evolving consensus."
    )

# --- Main Runner ---
async def main(topics: list[str], agent_count: int, topic_name: str):
    topic_str = ", ".join(topics)
    print(f"\n--- Committee Debate on: {topic_str} ---")

    prior_reports = load_previous_reports(topic_name)
    session_service = InMemorySessionService()
    await session_service.create_session(app_name="agents", user_id="user_123", session_id="session_456")

    coordinator = build_debate_coordinator(agent_count, topic_str, prior_reports)
    runner = Runner(agent=coordinator, app_name="agents", session_service=session_service)
    user_message = Content(parts=[{'text': topic_str}], role="user")

    agent_outputs = {}
    consensus_text = None

    async for event in runner.run_async(new_message=user_message, user_id="user_123", session_id="session_456"):
        author = event.author
        content = event.content.parts[0].text if event.content and event.content.parts else ""
        if author.startswith("DebateAgent_") or author == "ConsensusAgent":
            agent_outputs[author] = content
            if author == "ConsensusAgent":
                consensus_text = content

    full_report = f"Topics: {topic_str}\n" + "-"*40 + "\n\n"
    for agent_name, output in agent_outputs.items():
        full_report += f"\nAgent: {agent_name}\n" + "-"*20 + "\n" + output + "\n"
    full_report += "\n--- Final Consensus ---\n" + (consensus_text or "No consensus generated.")

    save_committee_report(topic_name, full_report)
    print("\n🧾 Final Consensus:\n" + (consensus_text or "No consensus generated."))

# --- CLI Entry Point ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python committee_debate.py <topic1> <topic2> ... [--agents N] [--topicname NAME]")
        sys.exit(1)

    args = sys.argv[1:]
    agent_count = 3
    topic_name = "MyTopic001"

    if "--agents" in args:
        idx = args.index("--agents")
        try:
            agent_count = int(args[idx + 1])
            args.pop(idx + 1)
            args.pop(idx)
        except:
            print("Error: '--agents' must be followed by an integer.")
            sys.exit(1)

    if "--topicname" in args:
        idx = args.index("--topicname")
        try:
            topic_name = args[idx + 1]
            args.pop(idx + 1)
            args.pop(idx)
        except:
            print("Error: '--topicname' must be followed by a string.")
            sys.exit(1)

    topics = args

    try:
        asyncio.run(main(topics, agent_count, topic_name))
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)