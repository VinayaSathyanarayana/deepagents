import sys
import asyncio
import os
import glob
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Import specific exceptions for retry logic
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content

# --- Load Environment Variables ---
load_dotenv()

# --- (NEW) Get API Keys in Sequence ---
def get_api_keys() -> list[str]:
    """
    Finds all GOOGLE/GEMINI API keys from environment variables.
    
    Returns keys in order:
    1. GEMINI_API_KEY / GOOGLE_API_KEY
    2. Numbered keys (GEMINI_API_KEY_001, GOOGLE_API_KEY002, etc.) sorted.
    """
    keys = {}
    for key, value in os.environ.items():
        if key.startswith("GEMINI_API_KEY") or key.startswith("GOOGLE_API_KEY"):
            keys[key] = value

    base_keys = []
    numbered_keys = []

    # Prioritize base keys
    if "GEMINI_API_KEY" in keys:
        base_keys.append(keys.pop("GEMINI_API_KEY"))
    if "GOOGLE_API_KEY" in keys:
        base_keys.append(keys.pop("GOOGLE_API_KEY"))

    # Sort remaining numbered keys by name
    sorted_numbered = sorted(keys.items())
    numbered_keys = [v for k, v in sorted_numbered]

    all_keys = base_keys + numbered_keys
    if all_keys:
        print(f"Found {len(all_keys)} API keys to try.")
    return all_keys

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
        model="gemini-1.5-pro", # Using 1.5 Pro, update if needed
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
        model="gemini-1.5-pro", # Using 1.5 Pro, update if needed
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

    print("Running agents...")
    async for event in runner.run_async(new_message=user_message, user_id="user_123", session_id="session_456"):
        # (ENHANCEMENT 1) Add try/except for robust event processing
        try:
            author = event.author
            content = ""
            if event.content and event.content.parts and event.content.parts[0].text:
                content = event.content.parts[0].text
            
            if author.startswith("DebateAgent_"):
                print(f"-> Received view from {author}")
                agent_outputs[author] = content
            elif author == "ConsensusAgent":
                print("-> Received final consensus")
                consensus_text = content # Store separately for final report
                agent_outputs[author] = content # Also store in dict
            elif author == "User":
                pass # Ignore user input
            # else:
                # print(f"-> Received other event from {author}") # Optional: for debugging tools

        except Exception as e:
            print(f"⚠️ Error processing event: {e}\nEvent: {event}")

    print("Agent run finished.")
    
    # Build the final report
    full_report = f"Topics: {topic_str}\n" + "-"*40 + "\n\n"
    for agent_name, output in agent_outputs.items():
        if agent_name == "ConsensusAgent":
            continue # We add this last, from the consensus_text variable
        full_report += f"\nAgent: {agent_name}\n" + "-"*20 + "\n" + output + "\n"
    
    full_report += "\n--- Final Consensus ---\n" + (consensus_text or "No consensus generated.")

    save_committee_report(topic_name, full_report)
    print("\n🧾 Final Consensus:\n" + (consensus_text or "No consensus generated."))

# --- CLI Entry Point ---
if __name__ == "__main__":
    args = sys.argv[1:]
    agent_count = 3
    topic_name = "MyTopic001"

    # --- Arg Parsing ---
    if "--agents" in args:
        idx = args.index("--agents")
        try:
            agent_count = int(args[idx + 1])
            del args[idx:idx + 2]
        except:
            print("Error: '--agents' must be followed by an integer.")
            sys.exit(1)

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

    topics = args
    if not topics:
        print("Usage: python committee_debate.py <topic1> <topic2> ... [--agents N] [--topicname NAME]")
        sys.exit(1)

    # --- (ENHANCEMENT 2) API Key Rotation and Retry Loop ---
    api_keys = get_api_keys()
    if not api_keys:
        print("❌ Error: No API keys found.")
        print("Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file or environment.")
        sys.exit(1)

    key_index = 0
    success = False
    while key_index < len(api_keys):
        current_key = api_keys[key_index]
        print(f"\n--- Attempt {key_index + 1}/{len(api_keys)}: Using key ending in '...{current_key[-4:]}' ---")
        
        # Set the environment variable for google-genai to pick up
        os.environ["GOOGLE_API_KEY"] = current_key

        try:
            asyncio.run(main(topics, agent_count, topic_name))
            success = True
            print("\n✅ Run completed successfully.")
            break # Exit loop on success

        except (ResourceExhausted, InternalServerError, ServiceUnavailable) as e:
            print(f"⚠️ API Error (Retryable) with key index {key_index}: {type(e).__name__}")
            key_index += 1
            if key_index < len(api_keys):
                print("Retrying with next key...")
            else:
                print("❌ All API keys failed or are rate-limited.")
        
        except Exception as e:
            # Catch all other non-retryable errors
            print(f"❌ An unrecoverable error occurred: {type(e).__name__} - {e}")
            traceback.print_exc()
            break # Exit loop, do not retry

    if not success:
        print("\nFailed to complete the run after trying all keys.")
        sys.exit(1)