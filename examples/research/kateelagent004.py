import os
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow

# Load Gemini API key from .env
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")  # Ensure .env file contains GOOGLE_API_KEY=your_actual_key

llm = GoogleGenAI(model="gemini-2.5-pro", api_key=API_KEY)

# Define tool (logic stub; actual web tools can be added with llama-index-tools-google)
def record_notes(topic):
    # Simulate research; in production, use search tool or GoogleWebToolSpec
    return f"Collected key facts and notes on: {topic}"

def write_report(notes):
    return f"# Blog Report\n\n{notes}\n\nWritten by Gemini-2.5-Pro."

def review_report(report):
    # Gemini model reviews internally for this simple demo
    return f"Review: This report is clear, complete, and well-written. Approved!"

# Define agents with system prompts
research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Searches the web for information on a given topic and records notes.",
    system_prompt="You are the ResearchAgent. Search for facts, statistics, and news about the topic. Once done, hand off to WriteAgent.",
    llm=llm,
    tools=[record_notes],
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Writes a report based on research notes.",
    system_prompt="You are the WriteAgent. Compose a report in markdown from research notes. Hand off to ReviewAgent for feedback.",
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Reviews the report for clarity and correctness.",
    system_prompt="You are the ReviewAgent. Review the report and provide feedback.",
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent", "ResearchAgent"],
)

# Build agent workflow
agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)

def run_multi_agent_workflow(topic):
    input_event = topic
    result_state = agent_workflow.run(user_msg=input_event)
    # For simple demo, extract states directly
    print("Report Content:\n", result_state["report_content"])
    print("\n------------\nFinal Review:\n", result_state["review"])

if __name__ == "__main__":
    topic = "Recent advancements in AI reasoning"
    run_multi_agent_workflow(topic)
