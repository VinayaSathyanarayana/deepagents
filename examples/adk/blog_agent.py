import sys
import asyncio
import os
from dotenv import load_dotenv
from typing import AsyncGenerator

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search  # Built-in Google Search tool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content

# --- Load Environment Variables ---
# This loads the GOOGLE_API_KEY and GOOGLE_GENAI_USE_VERTEXAI from the.env file
load_dotenv()

# --- 1. Define Specialist Agents ---

# Agent 1: The Researcher
# This agent's job is to take a topic, use Google Search, and output findings.
ResearchAgent = LlmAgent(
    model="gemini-2.5-pro",
    name="ResearchAgent",
    instruction="""
    You are a professional researcher. Your goal is to take a topic,
    use the google_search tool to find 3-5 key facts, statistics, or
    recent developments about it.
    
    Synthesize these findings into a concise, bulleted list. This list
    will be the *only* information passed to the writer.
    """,
    description="Researches a topic using Google Search and provides a summary.",
    tools=[google_search],
    # This is the critical part: the agent's final output will be saved
    # into the session state under the key 'research_findings'.
    output_key="research_findings",
)

# Agent 2: The Blog Writer
# This agent does not search. It only writes based on the researcher's findings.
BlogWriterAgent = LlmAgent(
    model="gemini-2.5-pro",
    name="BlogWriterAgent",
    instruction="""
    You are a professional blog post writer. You will be given a list of
    research findings. Your job is to transform these findings into an
    engaging, well-structured blog post.
    
    The post must have:
    1. A compelling title.
    2. An introductory paragraph.
    3. The main body, which expands on the research findings.
    4. A concluding paragraph.
    
    Here are the research findings you must use:
    {research_findings}
    """,
    description="Writes a blog post from a set of research findings.",
    # This agent has no tools. It only processes the input from its instruction.
)

# --- 2. Define the Coordinator Agent (The Team) ---

# This SequentialAgent acts as the team manager.
# It ensures Agent 1 runs, then Agent 2 runs, passing the data between them.
BlogCoordinator = SequentialAgent(
    name="BlogCoordinator",
    sub_agents=,
    description="Coordinates the research and writing process for a blog post.",
)

# --- 3. Define the Programmatic Runner ---

async def main(topic: str):
    """
    Asynchronously runs the agent workflow.
    """
    print(f"--- Starting Blog Generation for Topic: '{topic}' ---")

    # A. Setup Runner Dependencies
    session_service = InMemorySessionService()  # Manages agent memory/state
    app_name = "blog_agent_app"
    user_id = "user_123"
    session_id = "session_456"

    # B. Initialize the Runner
    # The Runner is the main entry point for running agents programmatically.
    runner = Runner(
        agent=BlogCoordinator,  # We run the coordinator, not the individual agents
        app_name=app_name,
        session_service=session_service,
    )

    # C. Create a Session Context
    # This ensures the agents have a shared state to pass data (research_findings)
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    # D. Execute the Agent Workflow
    # We pass the user's topic as the first query.
    # The 'query' (topic) goes to the first agent in the sequence (ResearchAgent).
    print(f"\n Passing topic to {ResearchAgent.name} for research...")
    
    # The runner.run() method handles the entire sequential flow.
    llm_response = await runner.run(
        query=topic,  # The topic from the command line
        user_id=user_id,
        session_id=session_id
    )
    
    print(f"\n {BlogWriterAgent.name} has completed the blog.")

    # E. Print the Final Result
    # The final_response_text will be the output of the *last* agent
    # in the sequence (BlogWriterAgent).
    print("\n--- Generated Blog Post ---")
    print(llm_response.final_response_text)
    print("-----------------------------\n")
    print("Workflow complete.")


# --- 4. Define the Command-Line Entry Point ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide a topic as a command-line argument.")
        print('Example: python blog_agent.py "The Future of Quantum Computing"')
        sys.exit(1)

    # Combine all arguments into a single topic string
    topic_from_cli = " ".join(sys.argv[1:])

    try:
        # asyncio.run() is the standard way to run an async main function
        # from a synchronous script.
        asyncio.run(main(topic_from_cli))
    except Exception as e:
        print(f"An error occurred during the agent workflow: {e}")
        print("God is great")
        sys.exit(1)