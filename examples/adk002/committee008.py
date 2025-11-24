
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
from google.adk.models.lite_llm import LiteLlm
import requests

# --- Load Environment Variables ---
load_dotenv()

# --- Tavily Search Integration ---
def tavily_search(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "[Warning] Tavily API key missing. No search context added."
    url = "https://api.tavily.com/search"
    payload = {"query": query, "api_key": api_key, "max_results": 5}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            results = [item.get("content", "") for item in data.get("results", [])]
            return "\n".join(results)
        else:
            return f"[Tavily Error] Status: {response.status_code}"
    except Exception as e:
        return f"[Tavily Exception] {e}"

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

# --- Save New Report ---
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

# --- Hybrid Debate Agent ---
def create_debate_agent(index: int, topic_str: str, model: str) -> LlmAgent:
    tools_list = []
    search_context = ""
    model_name = str(model)

    # Hybrid logic
    if model_name.startswith("gemini"):
        tools_list = [google_search]  # Gemini supports Google Search
    elif any(model_name.startswith(p) for p in ["anthropic/", "ollama/"]):
        search_context = tavily_search(topic_str)  # Claude/Ollama use Tavily

    return LlmAgent(
        model=model,
        name=f"DebateAgent_{index}",
        instruction=f"""
You are DebateAgent_{index}, a domain expert with a unique perspective.
Your task is to analyze the following management challenge:
{topic_str}

Additional context from web search:
{search_context}

Present your viewpoint clearly, citing 3–5 key arguments or insights.
Be constructive and persuasive, anticipating counterpoints.
""",
        description=f"Debates the topic from perspective {index}.",
        tools=tools_list,
        output_key=f"agent_{index}_view"
    )

# --- Consensus Agent ---
def create_consensus_agent(agent_count: int, prior_reports: str, new_inputs: str, model: str) -> LlmAgent:
    views = "\n".join([f"{{agent_{i}_view}}" for i in range(agent_count)])
    return LlmAgent(
        model=model,
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

# --- Coordinator ---
def build_debate_coordinator(agent_count: int, topic_str: str, prior_reports: str, model: str) -> SequentialAgent:
    debate_agents = [create_debate_agent(i, topic_str, model) for i in range(agent_count)]
    consensus_agent = create_consensus_agent(agent_count, prior_reports, topic_str, model)
    return SequentialAgent(
        name="DebateCoordinator",
        sub_agents=debate_agents + [consensus_agent],
        description="Coordinates expert debate and evolving consensus."
    )

# --- Main Runner ---
async def main(topics: list[str], agent_count: int, topic_name: str, use_genai: str):
    model_map = {
        "claude": "anthropic/claude-3-5-sonnet-20241022",
        "claude-opus": "anthropic/claude-3-opus-20240229",
        "gemini": "gemini-3-pro-preview",
        "gemini-flash": "gemini-2.5-flash",
        "deepseek-local": "ollama/deepseek-coder:6.7b",
    }

    provider_lower = use_genai.lower()
    if provider_lower not in model_map:
        print(f"❌ Error: Unknown AI provider '{use_genai}'")
        print(f" Available providers: {', '.join(model_map.keys())}")
        sys.exit(1)

    model_str = model_map[provider_lower]
    if any(p in model_str for p in ["anthropic/", "ollama/"]):
        model = LiteLlm(model=model_str)
        print(f"🤖 Using {use_genai} AI via LiteLLM")
    else:
        model = model_str
        print(f"🤖 Using {use_genai} AI")

    print(f" Model: {model_str}")
    topic_str = ", ".join(topics)
    print(f"\n--- Committee Debate on: {topic_str} ---")

    prior_reports = load_previous_reports(topic_name)
    session_service = InMemorySessionService()
    await session_service.create_session(app_name="agents", user_id="user_123", session_id="session_456")

    coordinator = build_debate_coordinator(agent_count, topic_str, prior_reports, model)
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

    report_model_name = model_map[provider_lower]
    full_report = f"AI Provider: {use_genai.title()}\nModel: {report_model_name}\nTopics: {topic_str}\n" + "-"*40 + "\n\n"
    for agent_name, output in agent_outputs.items():
        full_report += f"\nAgent: {agent_name}\n" + "-"*20 + "\n" + output + "\n"
    full_report += "\n--- Final Consensus ---\n" + (consensus_text or "No consensus generated.")
    save_committee_report(topic_name, full_report)
    print("\n🧾 Final Consensus:\n" + (consensus_text or "No consensus generated."))

# --- CLI Entry Point ---
if __name__ == "__main__":
    args = sys.argv[1:]
    agent_count = 3
    topic_name = "MyTopic001"
    use_genai = "claude"

    if "--agents" in args:
        idx = args.index("--agents")
        try:
            agent_count = int(args[idx + 1])
            del args[idx:idx + 2]
        except:
            print("Error: '--agents' must be followed by an integer.")
            sys.exit(1)

    if "--topicname" in args or "--topic" in args:
        idx = args.index("--topicname") if "--topicname" in args else args.index("--topic")
        try:
            topic_name = args[idx + 1]
            del args[idx:idx + 2]
        except:
            print("Error: '--topicname' or '--topic' must be followed by a string.")
            sys.exit(1)

    if "--usegenai" in args:
        idx = args.index("--usegenai")
        try:
            use_genai = args[idx + 1]
            del args[idx:idx + 2]
        except (IndexError, ValueError):
            print("Error: '--usegenai' must be followed by a provider name.")
            sys.exit(1)

    topics = args
    if not topics:
        print("Usage: python committee008.py <topic1> <topic2> ... [--agents N] [--topicname NAME] [--usegenai PROVIDER]")
        sys.exit(1)

    try:
        asyncio.run(main(topics, agent_count, topic_name, use_genai))
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)
