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
# Ensure LiteLlm is available for non-native ADK models like those accessed via Ollama
from google.adk.models.lite_llm import LiteLlm 

# --- Enhanced Documentation ---
"""
Committee Debate System with Multi-LLM Support
==============================================

A debate system that uses multiple AI agents to analyze topics and generate
consensus reports. Supports cloud models (Claude, Gemini) and local models 
(DeepSeek via Ollama).

SETUP INSTRUCTIONS:
------------------
1. Install required packages:
    pip install google-adk python-dotenv litellm 
    # litellm is required for seamless integration of non-native models like Ollama

2. Set up Ollama and DeepSeek:
    a. Install and run Ollama: https://ollama.com/
    b. Pull the DeepSeek model: ollama pull deepseek-coder:6.7b

3. Set up API keys in .env file or environment (if using cloud models):
    ANTHROPIC_API_KEY=your-anthropic-key-here    # Required for Claude
    GOOGLE_API_KEY=your-google-key-here          # Required for Gemini

USAGE:
------
Basic usage (defaults to Claude 3.5 Sonnet):
  python committee_debate.py "topic1" "topic2"

Specify Local AI provider (DeepSeek via Ollama):
  python committee_debate.py "topic1" "topic2" --usegenai deepseek-local
  
Specify Cloud AI provider:
  python committee_debate.py "topic1" "topic2" --usegenai gemini

Advanced options:
  python committee_debate.py "market expansion" "cost reduction" \
    --agents 5 \
    --topicname StrategyQ4 \
    --usegenai deepseek-local

AVAILABLE AI PROVIDERS:
-----------------------
  claude          Claude 3.5 Sonnet (requires ANTHROPIC_API_KEY)
  claude-opus     Claude 3 Opus - most capable (requires ANTHROPIC_API_KEY)
  gemini          Gemini 3 Pro Preview
  gemini-flash    Gemini 2.5 Flash Experimental
  deepseek-local  DeepSeek Coder 6.7B via local Ollama instance (requires Ollama)

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
- Support for multiple AI providers (cloud and local via Ollama)

CHANGELOG:
----------
v2.1 - Added local Ollama support via LiteLLM for DeepSeek Coder.
v2.0 - Added multi-LLM support with Claude and Gemini.
"""
# --- End of Enhanced Documentation ---

# --- Load Environment Variables ---
load_dotenv()

# --- Load Previous Reports (Function remains unchanged) ---
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

# --- Save New Report with Incremental Filename (Function remains unchanged) ---
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

# --- Define Debate Agent (Function remains unchanged) ---
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

# --- Define Consensus Agent (Function remains unchanged) ---
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

# --- Define Coordinator Agent (Function remains unchanged) ---
def build_debate_coordinator(agent_count: int, topic_str: str, prior_reports: str, model: str) -> SequentialAgent:
    debate_agents = [create_debate_agent(i, topic_str, model) for i in range(agent_count)]
    consensus_agent = create_consensus_agent(agent_count, prior_reports, topic_str, model)
    return SequentialAgent(
        name="DebateCoordinator",
        sub_agents=debate_agents + [consensus_agent],
        description="Coordinates expert debate and evolving consensus."
    )

# --- Main Runner (Modified for Ollama/DeepSeek) ---
async def main(topics: list[str], agent_count: int, topic_name: str, use_genai: str):
    
    # 🌟 NEW: Added deepseek-local entry
    model_map = {
        # Claude via LiteLLM (requires ANTHROPIC_API_KEY)
        "claude": "anthropic/claude-3-5-sonnet-20241022",
        "claude-opus": "anthropic/claude-3-opus-20240229",
        "claude-haiku": "anthropic/claude-3-haiku-20240307",
        # Gemini (native to ADK)
        "gemini": "gemini-3-pro-preview",
        "gemini-flash": "gemini-2.5-flash",
        # 🌟 NEW: DeepSeek via LiteLLM (Ollama provider)
        # Note: 'ollama/deepseek-coder:6.7b' uses the litellm format for Ollama
        "deepseek-local": "ollama/deepseek-coder:6.7b", 
    }
    
    provider_lower = use_genai.lower()
    if provider_lower not in model_map:
        print(f"❌ Error: Unknown AI provider '{use_genai}'")
        print(f"   Available providers: {', '.join(model_map.keys())}")
        # Updated setup instructions for clarity
        print(f"\n   Make sure you have the required setup:")
        print(f"   - Claude models require: ANTHROPIC_API_KEY")
        print(f"   - Gemini models require: GOOGLE_API_KEY (or ADK default config)")
        print(f"   - deepseek-local requires: Ollama running with 'deepseek-coder:6.7b' pulled.")
        sys.exit(1)
    
    model_str = model_map[provider_lower]
    
    # Wrap LiteLLM-based models (including Claude, OpenAI, and now Ollama)
    if any(p in model_str for p in ["anthropic/", "openai/", "groq/", "ollama/"]):
        
        # 🌟 NEW: Special handling for Ollama to set the base URL if needed
        if model_str.startswith("ollama/"):
            # LiteLLM automatically uses the default http://localhost:11434, 
            # but you could set LITELLM_OLLAMA_BASE_URL env var if your setup differs.
            print(f"💡 Running {model_str.split('/')[-1]} locally. Ensure Ollama is running on http://localhost:11434.")
        
        model = LiteLlm(model=model_str)
        print(f"🤖 Using {use_genai} AI via LiteLLM")
        print(f"   Model: {model_str}")
    else:
        # For native ADK models (Gemini)
        model = model_str
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

    # Get the *original* model name string for the report, handling LiteLlm
    report_model_name = model_map[provider_lower] 
    
    full_report = f"AI Provider: {use_genai.title()}\nModel: {report_model_name}\nTopics: {topic_str}\n" + "-"*40 + "\n\n"
    for agent_name, output in agent_outputs.items():
        full_report += f"\nAgent: {agent_name}\n" + "-"*20 + "\n" + output + "\n"
    full_report += "\n--- Final Consensus ---\n" + (consensus_text or "No consensus generated.")

    save_committee_report(topic_name, full_report)
    print("\n🧾 Final Consensus:\n" + (consensus_text or "No consensus generated."))

# --- CLI Entry Point (Modified to update instructions) ---
if __name__ == "__main__":
    # ... (CLI parsing logic remains mostly the same) ...
    args = sys.argv[1:]
    agent_count = 3
    topic_name = "MyTopic001"
    use_genai = "claude" 

    # --- (CLI argument parsing code remains unchanged) ---
    
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
            print("Error: '--usegenai' must be followed by a provider name (e.g., 'claude' or 'deepseek-local').")
            sys.exit(1)

    # Remaining args are actual debate topics
    topics = args

    if not topics:
        print("Usage: python committee_debate.py <topic1> <topic2> ... [--agents N] [--topicname NAME] [--usegenai PROVIDER]")
        print("\nOptions:")
        print("   --agents N          Number of debate agents (default: 3)")
        print("   --topicname NAME    Name for report files (default: MyTopic001)")
        print("   --usegenai PROVIDER AI provider to use (default: claude)")
        print("\nAvailable Providers (v2.1):")
        print("   claude              Claude 3.5 Sonnet (requires ANTHROPIC_API_KEY)")
        print("   claude-opus         Claude 3 Opus (requires ANTHROPIC_API_KEY)")
        print("   gemini              Gemini 3 Pro Preview")
        print("   gemini-flash        Gemini 2.0 Flash Experimental")
        print("   deepseek-local      DeepSeek Coder 6.7B via local Ollama instance")
        print("\nEnvironment Variables Required:")
        print("   ANTHROPIC_API_KEY   For Claude models")
        print("   GOOGLE_API_KEY      For Gemini models")
        sys.exit(1)
        
    # --- (End of CLI argument parsing code) ---

    try:
        asyncio.run(main(topics, agent_count, topic_name, use_genai))
    except KeyError as e:
        print(f"\n❌ Configuration Error: {e}")
        print(f"   This usually means an API key is missing or the Ollama model is not pulled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)