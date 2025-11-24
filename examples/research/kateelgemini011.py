import os
import argparse
from typing import Literal

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

# --- Environment Setup -------------------------------------------------
# Make sure your TAVILY_API_KEY and GOOGLE_API_KEY environment variables
# are set in your shell before running this script.
# export TAVILY_API_KEY="your_tavily_key"
# export GOOGLE_API_KEY="your_google_key"
# -----------------------------------------------------------------------

# It's best practice to initialize the client once and reuse it.
try:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
except KeyError:
    print("Error: TAVILY_API_KEY environment variable not set.")
    exit(1)

try:
    # Initialize the Google Gemini model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
except Exception as e:
    print(f"Error initializing Google model: {e}")
    print("Please ensure GOOGLE_API_KEY is set and valid.")
    exit(1)


# --- Tool Definition ---------------------------------------------------

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

sub_research_prompt = """You are a dedicated researcher. Your job is to conduct research based on the users questions.

Conduct thorough research and then reply to the user with a detailed answer to their question

only your FINAL answer will be passed on to the user. They will have NO knowledge of anything except your final message, so your final report should be your final message!"""

research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "system_prompt": sub_research_prompt,
    "tools": [internet_search],
}

sub_critique_prompt = """You are a dedicated editor. You are being tasked to critique a report.

You can find the report at `final_report.md`.

You can find the question/topic for this report at `question.txt`.

The user may ask for specific areas to critique the report in. Respond to the user with a detailed critique of the report. Things that could be improved.

You can use the search tool to search for information, if that will help you critique the report

Do not write to the `final_report.md` yourself.

Things to check:
- Check that each section is appropriately named
- Check that the report is written as you would find in an essay or a textbook - it should be text heavy, do not let it just be a list of bullet points!
- Check that the report is comprehensive. If any paragraphs or sections are short, or missing important details, point it out.
- Check that the article covers key areas of the industry, ensures overall understanding, and does not omit important parts.
- Check that the article deeply analyzes causes, impacts, and trends, providing valuable insights
- Check that the article closely follows the research topic and directly answers questions
- Check that the article has a clear structure, fluent language, and is easy to understand.
"""

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "system_prompt": sub_critique_prompt,
}


# --- Main Agent Prompt -----------------------------------------------

research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

The first thing you should do is to write the original user question to `question.txt` so you have a record of it.

Use the research-agent to conduct deep research. It will respond to your questions/topics with a detailed answer.

When you think you enough information to write a final report, write it to `final_report.md`

You can call the critique-agent to get a critique of the final report. After that (if needed) you can do more research and edit the `final_report.md`
You can do this however many times you want until are you satisfied with the result.

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

Here are instructions for writing the final report:

<report_instructions>

CRITICAL: Make sure the answer is written in the same language as the human messages! If you make a todo plan - you should note in the plan what language the report should be in so you dont forget!
Note: the language the report should be in is the language the QUESTION is in, not the language/country that the question is ABOUT.

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1. intro
2. overview of topic A
3. overview of topic B
4. comparison between A and B
5. conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1. list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1. item 1
2. item 2
3. item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1. overview of topic
2. concept 1
3. concept 2
4. concept 3
5. conclusion

If you think you can answer the question with a single section, you can do that too!
1. answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
</report_instructions>

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""


# --- Main Execution --------------------------------------------------

def main():
    # 1. Setup to get research topic from command line
    parser = argparse.ArgumentParser(
        description="Run a deep research agent on a given topic."
    )
    parser.add_argument(
        "topic", type=str, help="The research topic you want the agent to write about."
    )
    args = parser.parse_args()
    research_topic = args.topic

    # 2. Define and create the output directory
    output_dir = "research_output"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Configure the agent's filesystem to write to our output directory
    # The agent's `write_file` tool will now write directly to the `research_output` folder.
    fs_backend = FilesystemBackend(root_dir=output_dir, virtual_mode=True)

    print(f"🚀 Initializing Deep Agent with Gemini 2.5 Pro...")

    # Create the agent
    agent = create_deep_agent(
        model=model,  # Use the Gemini model
        tools=[internet_search],
        system_prompt=research_instructions,
        subagents=[critique_sub_agent, research_sub_agent],
        backend=fs_backend,  # <-- This is the enhancement to save files
    )

    print(f"🧠 Starting research on: \"{research_topic}\"")
    print(f"📂 Output files will be saved to the '{output_dir}' directory.")

    # 4. Run the agent
    # The input is a list of messages. We start with the user's topic.
    messages = [{"role": "user", "content": research_topic}]

    # Invoke the agent and stream the final output
    # The agent will perform all its steps (planning, searching, writing to files)
    # and we will just print its final message.
    try:
        final_response = agent.invoke({"messages": messages})
        
        # The agent's final message (e.g., "I have completed the report.")
        if "messages" in final_response and final_response["messages"]:
            print("\n--- Agent's Final Message ---")
            print(final_response["messages"][-1].content)

        print("\n---")
        print(f"✅ Research complete!")
        print(f"📄 Your report is available at: {output_dir}/final_report.md")

    except Exception as e:
        print(f"\n--- 🛑 An error occurred during agent execution ---")
        print(e)
        print("Please check your API keys, network connection, and tool definitions.")


if __name__ == "__main__":
    main()