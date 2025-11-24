"""
Committee Debate System with Multi-LLM Support
==============================================

A debate system that uses multiple AI agents to analyze topics and generate
consensus reports. Supports Claude (via Anthropic API) and Gemini models.

SETUP INSTRUCTIONS:
------------------
1. Install required packages:
   pip install google-adk python-dotenv

2. Set up API keys in .env file or environment:
   ANTHROPIC_API_KEY=your-anthropic-key-here    # Required for Claude
   GOOGLE_API_KEY=your-google-key-here          # Required for Gemini (if not using ADK defaults)

USAGE:
------
Basic usage (defaults to Claude 3.5 Sonnet):
  python committee_debate.py "topic1" "topic2"

Specify AI provider:
  python committee_debate.py "topic1" "topic2" --usegenai claude
  python committee_debate.py "topic1" "topic2" --usegenai gemini

Advanced options:
  python committee_debate.py "market expansion" "cost reduction" \
    --agents 5 \
    --topicname StrategyQ4 \
    --usegenai claude-opus

AVAILABLE AI PROVIDERS:
-----------------------
  claude          Claude 3.5 Sonnet (requires ANTHROPIC_API_KEY)
  claude-opus     Claude 3 Opus - most capable (requires ANTHROPIC_API_KEY)
  claude-haiku    Claude 3 Haiku - fastest (requires ANTHROPIC_API_KEY)
  gemini          Gemini 3 Pro Preview
  gemini-flash    Gemini 2.0 Flash Experimental

COMMAND LINE OPTIONS:
--------------------
  --agents N          Number of debate agents (default: 3)
  --topicname NAME    Name for report files (default: MyTopic001)
  --usegenai PROVIDER AI provider to use (default: claude)

OUTPUT:
-------
Creates incremental report files: {topicname}.report001.txt, .report002.txt, etc.
Each report includes:
  - AI provider and model used
  - Individual agent viewpoints
  - Final consensus synthesis
  - Historical context from previous reports

FEATURES:
---------
- Multi-agent debate system with configurable number of agents
- Incremental report saving with version history
- Previous report integration for evolving analysis
- Web search capability for agents
- Support for multiple AI providers (easily extensible)

ADDING NEW AI PROVIDERS:
------------------------
To add support for new AI providers (GPT-4, Llama, etc.), update the model_map
dictionary in the main() function:

  model_map = {
      "claude": "anthropic/claude-3-5-sonnet-20241022",
      "gpt4": "openai/gpt-4-turbo",              # Add OpenAI
      "llama": "groq/llama-3-70b",               # Add Groq/Llama
  }

For LiteLLM-based providers, ensure the model string includes the provider prefix
(e.g., "anthropic/", "openai/", "groq/").

CHANGELOG:
----------
v2.0 - Added multi-LLM support with Claude and Gemini
     - Made AI provider configurable via --usegenai parameter
     - Added LiteLLM integration for Anthropic models
     - Improved error handling and user feedback
     - Added comprehensive documentation

v1.0 - Initial version with Gemini support only
"""

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
def create_debate_agent(index: int, topic_str: str, model: str) -> LlmAgent:
    return LlmAgent(
        model=model,
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

# --- Define Coordinator Agent ---
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
    # Select model based on provider
    # Note: Claude requires ANTHROPIC_API_KEY environment variable
    # Gemini works directly with ADK
    model_map = {
        # Claude via LiteLLM (requires ANTHROPIC_API_KEY)
        "claude": "anthropic/claude-3-5-sonnet-20241022",
        "claude-opus": "anthropic/claude-3-opus-20240229",
        "claude-haiku": "anthropic/claude-3-haiku-20240307",
        "claude-3-5-haiku": "anthropic/claude-3-5-haiku",
        # Gemini (native to ADK)
        "gemini": "gemini-3-pro-preview",
        "gemini-flash": "gemini-2.5-flash",
        
        # Vertex AI Claude (requires GCP project setup)
        # "claude-vertex": "vertex_ai/claude-3-7-sonnet@20250219",
        
        # Future providers can be added here:
        # "gpt4": "openai/gpt-4-turbo",
        # "llama": "groq/llama-3-70b",
    }
    
    provider_lower = use_genai.lower()
    if provider_lower not in model_map:
        print(f"❌ Error: Unknown AI provider '{use_genai}'")
        print(f"   Available providers: {', '.join(model_map.keys())}")
        print(f"\n   Make sure you have the required API keys:")
        print(f"   - Claude models require: ANTHROPIC_API_KEY")
        print(f"   - Gemini models require: GOOGLE_API_KEY (or ADK default config)")
        sys.exit(1)
    
    model = model_map[provider_lower]
    
    # Wrap Claude and OpenAI models with LiteLLM for ADK compatibility
    if "anthropic/" in model or "openai/" in model or "groq/" in model:
        model = LiteLlm(model=model)
        print(f"🤖 Using {use_genai} AI via LiteLLM")
        print(f"   Model: {model_map[provider_lower]}")
    else:
        print(f"🤖 Using {use_genai} AI")
        print(f"   Model: {model}")
    
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

    full_report = f"AI Provider: {use_genai.title()}\nModel: {model_map[provider_lower]}\nTopics: {topic_str}\n" + "-"*40 + "\n\n"
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
    use_genai = "claude"  # Claude is default

    # Extract --agents
    if "--agents" in args:
        idx = args.index("--agents")
        try:
            agent_count = int(args[idx + 1])
            del args[idx:idx + 2]
        except:
            print("Error: '--agents' must be followed by an integer.")
            sys.exit(1)

    # Extract --topicname or --topic
    if "--topicname" in args or "--topic" in args:
        if "--topicname" in args:
            idx = args.index("--topicname")
        else:
            idx = args.index("--topic")
        try:
            topic_name = args[idx + 1]
            del args[idx:idx + 2]
        except:
            print("Error: '--topicname' or '--topic' must be followed by a string.")
            sys.exit(1)

    # Extract --usegenai flag with provider name
    if "--usegenai" in args:
        idx = args.index("--usegenai")
        try:
            use_genai = args[idx + 1]
            del args[idx:idx + 2]
        except (IndexError, ValueError):
            print("Error: '--usegenai' must be followed by a provider name (e.g., 'claude' or 'gemini').")
            sys.exit(1)

    # Remaining args are actual debate topics
    topics = args

    if not topics:
        print("Usage: python committee_debate.py <topic1> <topic2> ... [--agents N] [--topicname NAME] [--usegenai PROVIDER]")
        print("\nOptions:")
        print("  --agents N          Number of debate agents (default: 3)")
        print("  --topicname NAME    Name for report files (default: MyTopic001)")
        print("  --usegenai PROVIDER AI provider to use (default: claude)")
        print("\nAvailable Providers:")
        print("  claude              Claude 3.5 Sonnet (requires ANTHROPIC_API_KEY)")
        print("  claude-opus         Claude 3 Opus (requires ANTHROPIC_API_KEY)")
        print("  claude-haiku        Claude 3 Haiku (requires ANTHROPIC_API_KEY)")
        print("  gemini              Gemini 3 Pro Preview")
        print("  gemini-flash        Gemini 2.0 Flash Experimental")
        print("\nEnvironment Variables Required:")
        print("  ANTHROPIC_API_KEY   For Claude models")
        print("  GOOGLE_API_KEY      For Gemini models (if not using ADK defaults)")
        sys.exit(1)

    try:
        asyncio.run(main(topics, agent_count, topic_name, use_genai))
    except KeyError as e:
        print(f"\n❌ Configuration Error: {e}")
        print(f"   This usually means an API key is missing.")
        print(f"   Please check your .env file or environment variables:")
        print(f"   - ANTHROPIC_API_KEY for Claude models")
        print(f"   - GOOGLE_API_KEY for Gemini models")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print(f"\n   Common issues:")
        print(f"   1. Missing API key - check .env file or environment")
        print(f"   2. Invalid model name - verify provider supports the model")
        print(f"   3. Network issues - check internet connection")
        print(f"   4. API quota exceeded - check your API usage limits")
        sys.exit(1)