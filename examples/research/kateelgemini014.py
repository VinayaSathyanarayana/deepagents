import os
import argparse
from typing import Literal

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

# We need the specific exception from Google's library to catch it
from google.api_core.exceptions import ResourceExhausted
# tenacity is used by LangChain for retries
from tenacity import wait_exponential, stop_after_attempt, RetryError

# --- Environment Setup -------------------------------------------------
# (Same as before)
try:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
except KeyError:
    print("Error: TAVILY_API_KEY environment variable not set.")
    exit(1)

try:
    # --- MODEL INITIALIZATION WITH RETRY ---
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro").with_retry(
        retry_on_exceptions=(ResourceExhausted,),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    print("🤖 Model initialized with automatic retry logic for rate limits.")

except Exception as e:
    print(f"Error initializing Google model: {e}")
    print("Please ensure GOOGLE_API_KEY is set and valid.")
    exit(1)


# --- Tool Definition ---------------------------------------------------
# (Same as before)
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs


# --- Sub-agent Prompts -----------------------------------------------
# (All prompts are the same as before - omitted for brevity)

sub_research_prompt = """You are a dedicated researcher. Your job is to conduct research based on the users questions.
..."""

research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "system_prompt": sub_research_prompt,
    "tools": [internet_search],
}

sub_critique_prompt = """You are a dedicated editor. You are being tasked to critique a report.
..."""

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "system_prompt": sub_critique_prompt,
}


# --- Main Agent Prompt -----------------------------------------------
# (Same as before - omitted for brevity)
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.
..."""


# --- Main Execution (MODIFIED) ---------------------------------------

def main():
    # 1. Setup (Same as before)
    parser = argparse.ArgumentParser(
        description="Run a deep research agent on a given topic."
    )
    parser.add_argument(
        "topic", type=str, help="The research topic you want the agent to write about."
    )
    args = parser.parse_args()
    research_topic = args.topic

    # 2. Output Dir (Same as before)
    output_dir = "research_output"
    os.makedirs(output_dir, exist_ok=True)
    fs_backend = FilesystemBackend(root_dir=output_dir, virtual_mode=True)

    print(f"🚀 Initializing Deep Agent with Gemini 2.5 Pro...")

    # 3. Create Agent (Same as before)
    agent = create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt=research_instructions,
        subagents=[critique_sub_agent, research_sub_agent],
        backend=fs_backend,
    )

    print(f"🧠 Starting research on: \"{research_topic}\"")
    print(f"📂 Output files will be saved to the '{output_dir}' directory.")
    print("ℹ️  Note: If rate limits are hit, the agent will pause and retry automatically.")

    # 4. Run the agent (MODIFIED to use .stream() for detailed progress)
    messages = [{"role": "user", "content": research_topic}]

    try:
        # Keep track of messages we've already printed
        printed_message_ids = set()
        final_response_dict = None  # To store the final state

        # Use agent.stream() instead of agent.invoke()
        # The default stream_mode="values" gives us dicts
        # representing the agent's state.
        for chunk in agent.stream({"messages": messages}):
            # Store the latest state chunk
            final_response_dict = chunk

            # We are interested in the "messages" key, which holds the chat history
            if "messages" not in chunk:
                continue

            # Find any new messages we haven't printed yet
            new_messages = []
            for msg in chunk["messages"]:
                if msg.id not in printed_message_ids:
                    new_messages.append(msg)
                    printed_message_ids.add(msg.id)

            if not new_messages:
                continue  # No new messages in this chunk

            # Get the single newest message to print
            newest_message = new_messages[-1]
            
            print("\n" + "="*80)

            # --- Pretty Print Agent Steps ---

            if newest_message.type == "ai":
                print("🧠 Agent is thinking/acting...")
                if newest_message.tool_calls:
                    print("ACTION:")
                    for tool_call in newest_message.tool_calls:
                        # Prettier printing for known sub-agents and file ops
                        if tool_call['name'] == 'task:research-agent':
                            print(f"  - Delegating to Research Sub-Agent:")
                            print(f"    Topic: {tool_call['args'].get('question')}")
                        elif tool_call['name'] == 'task:critique-agent':
                            print(f"  - Delegating to Critique Sub-Agent:")
                            print(f"    Instructions: {tool_call['args'].get('question')}")
                        elif tool_call['name'] in ['write_file', 'edit_file']:
                            print(f"  - Calling Filesystem Tool: {tool_call['name']}")
                            print(f"    File: {tool_call['args'].get('path')}")
                        elif tool_call['name'] == 'internet_search':
                             print(f"  - Calling Tool: {tool_call['name']}")
                             print(f"    Query: {tool_call['args'].get('query')}")
                        else:
                            print(f"  - Calling tool: {tool_call['name']}")
                            print(f"  - Arguments: {tool_call['args']}")
                else:
                    # This is a final message or an interim chat message
                    print("AGENT MESSAGE:")
                    print(newest_message.content)

            elif newest_message.type == "tool":
                print(f"🛠️ Tool Executed: {newest_message.name}")
                # Truncate long tool outputs (like search results) for readability
                content_preview = str(newest_message.content)
                if len(content_preview) > 500:
                    content_preview = content_preview[:500] + "\n... (output truncated)"
                print(f"RESULT: {content_preview}")

            elif newest_message.type == "human":
                # This will print the initial user prompt at the start
                print(f"👤 User Input:")
                print(newest_message.content)

            print("="*80)
        
        # --- End of Stream ---

        # After the loop, print the final message from the last stored state
        if final_response_dict and "messages" in final_response_dict:
            print("\n--- Agent's Final Message ---")
            print(final_response_dict["messages"][-1].content)
        
        print("\n---")
        print(f"✅ Research complete!")
        print(f"📄 Your report is available at: {output_dir}/final_report.md")

    # (Same exception handling as before)
    except (ResourceExhausted, RetryError) as e:
        print(f"\n--- 🛑 FATAL: Rate Limit Error ---")
        print("The agent failed to complete after multiple retries due to persistent rate limiting.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n--- 🛑 An unexpected error occurred during agent execution ---")
        print(e)
        print("Please check your API keys, network connection, and tool definitions.")


if __name__ == "__main__":
    main()