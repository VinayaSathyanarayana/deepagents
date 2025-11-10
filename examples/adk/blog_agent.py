import sys
import asyncio
from dotenv import load_dotenv

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search  # Built-in Google Search tool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
# from google.genai.types import Content  <- Removed, not needed for this method

# --- Load Environment Variables ---
load_dotenv()

# --- 1. Define Specialist Agents ---

ResearchAgent = LlmAgent(
    model="gemini-2.5-pro",
    name="ResearchAgent",
    instruction="""
    You are a professional researcher. Your goal is to take the topic
    from the user's initial query, use the google_search tool to find
    3-5 key facts, statistics, or recent developments about it.
    
    Synthesize these findings into a concise, bulleted list. This list
    will be the *only* information passed to the writer.
    """,
    description="Researches a topic using Google Search and provides a summary.",
    tools=[google_search],
    output_key="research_findings",
)

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
)

# --- 2. Define the Coordinator Agent (The Team) ---

BlogCoordinator = SequentialAgent(
    name="BlogCoordinator",
    sub_agents=[ResearchAgent, BlogWriterAgent],
    description="Coordinates the research and writing process for a blog post.",
)

# --- 3. Define the Programmatic Runner ---

async def main(topic: str):
    """
    Asynchronously runs the agent workflow.
    """
    print(f"--- Starting Blog Generation for Topic: '{topic}' ---")

    # A. Setup Runner Dependencies
    session_service = InMemorySessionService()
    
    # Use the 'agents' app_name as inferred from the error log
    app_name = "agents" 
    user_id = "user_123"
    session_id = "session_456"

    # B. Initialize the Runner
    runner = Runner(
        agent=BlogCoordinator,
        app_name=app_name,
        session_service=session_service,
    )

    # C. Create a Session Context
    # FIXED: The topic is passed as 'initial_query' *during* session creation.
    print(f"\n Creating session {session_id} with initial topic...")
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        initial_query=topic  # This is the new, correct argument
    )

    # D. Execute the Agent Workflow
    print(f" Passing control to {ResearchAgent.name} for research...")
    
    # The run() method now *only* needs the session identifiers,
    # as the topic was already part of the session creation.
    llm_response = await runner.run(
        user_id=user_id,
        session_id=session_id
    )
    
    print(f"\n {BlogWriterAgent.name} has completed the blog.")

    # E. Print the Final Result
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

    topic_from_cli = " ".join(sys.argv[1:])

    try:
        asyncio.run(main(topic_from_cli))
    except Exception as e:
        print(f"An error occurred during the agent workflow: {e}")
        print("God is great")
        sys.exit(1)