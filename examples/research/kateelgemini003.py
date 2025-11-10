import os
from typing import Literal
from deepagents import create_deep_agent
from duckduckgo_search import DDGS
import google.generativeai as genai

# ✅ Configure Gemini 2.5 Pro
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# ✅ DuckDuckGo search tool with docstring
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """
    Run a DuckDuckGo search for the given query. Returns a list of search results.
    """
    with DDGS() as ddgs:
        return ddgs.text(query, max_results=max_results)

# ✅ Sub-agent: Research (Gemini injected)
research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in-depth questions.",
    "system_prompt": """You are a dedicated researcher. Conduct thorough research and reply with a detailed answer. Only your FINAL answer will be passed to the user.""",
    "tools": [internet_search],
    "llm": gemini_model
}

# ✅ Sub-agent: Critique (Gemini injected)
critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report.",
    "system_prompt": """You are a dedicated editor. Critique the report at `final_report.md` based on the topic in `question.txt`. Do not edit the file yourself. Provide detailed feedback on structure, clarity, and completeness.""",
    "llm": gemini_model
}

# ✅ Main agent instructions
research_instructions = """You are an expert researcher. Write the user question to `question.txt`. Use the research-agent to gather information. Write the final report to `final_report.md`. Use the critique-agent to improve it.

Final report must:
- Use markdown headings
- Include facts and citations
- Be in the same language as the question
- End with a Sources section listing [Title](URL)

Tool available:
- `internet_search`: Run a web search for a given query.
"""

# ✅ Create the DeepAgent (Gemini injected only via subagents)
agent = create_deep_agent(
    tools=[internet_search],
    system_prompt=research_instructions,
    subagents=[critique_sub_agent, research_sub_agent]
)

# ✅ Run the agent
response = agent.invoke({"input": "What are the latest trends in cybersecurity for financial institutions?"})
print(response)