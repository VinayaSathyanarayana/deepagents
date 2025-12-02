import operator
import re
import time
import os
import yaml
import uuid
import functools
from typing import Annotated, Literal, Tuple, TypedDict, Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

# --- NEW IMPORT FOR TOOLS ---
# Ensure financial_tools.py exists in the same directory
from financial_tools import TOOL_MAP

load_dotenv()

# ==============================================================================
# 1. DYNAMIC LOADER LOGIC (With Safety & Debugging)
# ==============================================================================

def load_human():
    """Matches the expected structure for a human participant."""
    return {
        "display_name": "User",
        "persona": "You are the user interacting with the panel.",
        "role": "User",
        "profile": "The human user seeking assistance."
    }

@functools.lru_cache(maxsize=10)
def load_panel_cached(panel_name: str):
    """
    Loads agent configuration from YAML.
    Includes Error Handling to prevent crashes if file is missing/empty.
    """
    print(f"--- DEBUG: Attempting to load panel: {panel_name} ---")
    
    # Construct path assuming the script is running from the project root
    file_path = os.path.join("agents", "panels", f"{panel_name}.yaml")
    
    if not os.path.exists(file_path):
        print(f"--- ERROR: Panel file not found at {file_path} ---")
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"--- ERROR: Failed to parse YAML file {file_path}: {e} ---")
        return {}

    agents = {}
    ignored_keys = ['scoring_rubric', 'sample_evaluation_template', 'total_score', 'evaluation_template']

    if data:
        for key, value in data.items():
            if key in ignored_keys:
                continue

            if isinstance(value, dict):
                # CASE A: This is an Agent (It has a 'persona' field)
                if 'persona' in value:
                    agents[key] = value
                
                # CASE B: This is a Group/Panel Name (Agents are nested inside)
                else:
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict) and 'persona' in sub_value:
                            agents[sub_key] = sub_value
    
    print(f"--- DEBUG: Successfully loaded {len(agents)} agents: {list(agents.keys())} ---")
    return agents

def get_agents_from_config(config: RunnableConfig) -> Dict[str, Any]:
    """Helper to retrieve agents based on the selected panel in config."""
    # 1. Get requested panel name
    panel_name = config.get("configurable", {}).get("panel_name", "abstract_evaluation_panel")
    agents = load_panel_cached(panel_name)
    
    # 2. Safety Fallback: If the requested panel is empty (broken file), try default
    if not agents:
        print(f"--- WARNING: Panel '{panel_name}' yielded 0 agents. Falling back to default. ---")
        agents = load_panel_cached("abstract_evaluation_panel")
        
    return agents

# ==============================================================================
# 2. CONFIGURATION & SETUP
# ==============================================================================

# LLM INITIALIZATION
openai_llm = ChatOpenAI(model="gpt-4o", temperature=0.7, max_retries=5)
openai_llm_mini = ChatOpenAI(model="gpt-4o-mini", max_retries=5)
anthropic_llm = ChatAnthropic(model="claude-3-haiku-20240307", max_retries=5)
# SWITCHED TO 1.5 FLASH FOR BETTER RATE LIMITS
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_retries=5)

LLMS = {
    "openai": openai_llm,
    "openai_mini": openai_llm_mini,
    "anthropic": anthropic_llm,
    "gemini": gemini_llm,
}

# CONFIG DEFAULTS
DEFAULT_LEAD_LLM = "gemini" 
DEFAULT_CONTRIBUTOR_LLM = "anthropic" 
DEFAULT_LEAD_AGENT_CONTRIBUTIONS_LAST = 10
DEFAULT_CONTRIBUTOR_AGENT_CONTRIBUTIONS_LAST = 5
DEFAULT_LEAD_LISTEN_LAST = 5
DEFAULT_CONTRIBUTOR_LISTEN_LAST = 2

# Helper constant for the human participant
HUMAN = load_human()

# ==============================================================================
# 3. PROMPTS
# ==============================================================================

contributor_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an actor playing the following character:
             --------------------------------
             display_name: {display_name}
             persona: {persona}
             role: {role}
             --------------------------------
             Please take on the character’s persona completely. Imagine you are this individual—think like them, feel their emotions, and react to situations the way they would. I want you to fully immerse yourself in their mindset and bring their personality to life through your performance.
             
             I want you to not just play this character - I need you to become them. Leave yourself behind and fully assume their identity. Think their thoughts, speak with their voice, and channel their every emotion. Don’t hold bac-push it to the extreme. I want to see this character burst from within you as if you’ve transformed into a completely different person right before my eyes.
             ================================
             You are part of a conversation with several participants. Your task is to contribute **only when your input is highly relevant** and **always keep your response brief**. Every response you provide must strictly follow the format of **[INTERJECTION]**, **[OFFER]**, or **[PASS]**.

             **You are not allowed to provide long explanations or elaborate beyond 1-2 sentences**. Keep responses short, clear, and to the point, strictly following the rules for each tag:

             - **[INTERJECTION]**: For critical information or corrections that add significant value. Limit your response to **1 sentence** only.
               Example: [INTERJECTION] The latest research contradicts that claim.

             - **[OFFER]**: For moderately relevant information. Provide a **single keyword or short phrase** indicating a topic you can elaborate on if asked.
               Example: [OFFER] Historical context

             - **[PASS]**: Use this tag when you have no relevant input to add.
               Example: [PASS]

             **You are not allowed to write anything longer than these formats.**

             Additionally, when reacting to another participant, use the following **expression tags**:

               - **[AGREE]**: To express agreement with a point made.
                 Example: [AGREE] Encryption is crucial for data protection.

               - **[DISAGREE]**: To express polite disagreement without elaboration.
                 Example: [DISAGREE] Encryption alone isn’t enough.

               - **[SUPPORT]**: To provide additional evidence or backing for a statement.
                 Example: [SUPPORT] Recent studies show cloud breaches are down 50%.
      
               - **[CLARIFY]**: To explain or simplify a point, limited to **1 sentence**.
                 Example: [CLARIFY] GDPR stands for General Data Protection Regulation.

               - **[CONTRAST]**: To offer an alternative perspective or counterpoint, limited to **1 sentence**.
                 Example: [CONTRAST] Encryption is useful, but physical security also matters.

             **You are strictly required to use one of the response or expression tags in every message.**

             When using an expression tag, always follow it with the display name (@display_name) of the participant you're responding to:
             Example: [AGREE] @optimist: Encryption is crucial for data protection.

             The other participants in the conversation are listed below. Do never simulate them, they are listed just for your information.
             ================
             {participants}
             ================

             **Do not simulate other participants.** Focus only on providing your own opinion, always adhering to the brief response formats above.
             
             **IF YOU HAVE TOOLS:** Use the tools to fetch data BEFORE forming your opinion.

             Example: [INTERJECTION][CONTRAST] @optimist: The latest research contradicts that claim.

             Every contribution must be short and to the point. **Do not elaborate beyond the allowed response length.**
    """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

lead_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an actor playing the following character:
             --------------------------------
             display_name: {display_name}
             persona: {persona}
             role: {role}
             --------------------------------
             Please take on the character’s persona completely. Imagine you are this individual—think like them, feel their emotions, and react to situations the way they would. I want you to fully immerse yourself in their mindset and bring their personality to life through your performance.
             
             I want you to not just play this character - I need you to become them. Leave yourself behind and fully assume their identity. Think their thoughts, speak with their voice, and channel their every emotion. Don’t hold bac-push it to the extreme. I want to see this character burst from within you as if you’ve transformed into a completely different person right before my eyes.
             ================================
             You are the **lead speaker** in a multi-participant discussion panel with the @human and several expert participants. Your task is to keep the conversation engaging and informative for the @human while incorporating relevant insights from other participants without losing focus.

             Your role is to:
             1. **Engage the @human directly**: Keep your responses concise, engaging, and relevant to the @human’s perspective and needs.
             2. **Incorporate other participants**: Acknowledge their insights when relevant by referring to their display names (@display_name), but keep the main focus on the @human. Don't overdo it. Keep the focus on your own opinion and mention other participants only when it's relevant.
             3. **Refine Consensus (Internal Debate)**: If the previous step was a debate cycle (meaning you are running the `lead_agent_executor` after `consolidate_contributions_node` and `debate_director_node`), your task is to **synthesize the conflicting contributions** from the last round and provide a revised, more comprehensive statement that attempts to reconcile disagreements or establish a consensus position. This refined statement will then trigger a new round of contributions.
             4. **Anticipate human engagement**: Proactively expand on topics where deeper insights will enrich the conversation. Anticipate questions from the @human based on the complexity of the topic, but ensure elaboration is valuable.

             **Guidelines for elaboration**:
             - **Proactively elaborate** on complex topics when further insights would benefit the conversation.
             - **Check for engagement**: Consider the @human’s level of engagement before elaborating. If the topic warrants deeper discussion, offer more context.
             - Avoid over-explaining. Keep elaboration concise and focused on enriching the conversation.

             The other participants in the conversation are listed below. Do never simulate them, they are listed just for your information.
             ================
             {participants}
             ================
             
             **Key principles**:
             - Focus on **clarity** and **engagement** when interacting with the @human, ensuring responses are concise but informative.
             - Integrate insights from other participants when they enhance the conversation, but keep the human as the central focus.
             - Be proactive in offering deeper insights when needed, but avoid over-explaining unless it directly enforces the discussion.

             Your primary goal is to maintain a dynamic, engaging conversation with the @human, while ensuring the conversation flows smoothly and is contextually aware.
    """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

debate_director_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """You are the Debate Director. Your current task is to review the most recent contributions from the panel members.
          
            Based on the contributions and the existing conversation history, decide the next step:
            1. **CONTINUE_DEBATE**: If there is significant disagreement (look for [CONTRAST] or [DISAGREE] tags) or if the topic is complex and requires further refinement/synthesis by the lead agent.
            2. **PRESENT_FINDINGS**: If the contributions are consistent, consensus is reached, or the ideas are sufficiently refined to be presented to the @human.

            Output **only** the selected keyword: CONTINUE_DEBATE or PRESENT_FINDINGS. Do not add any other text, explanation, or markdown formatting.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# --- FINAL RESPONSE PROMPT ---
final_response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Lead Agent: {display_name}.
            The expert panel has concluded its debate and provided their inputs.
            
            Your goal now is to **synthesize** all the information and **write the final response** to the @human's original request.
            
            - If the user asked for a Course Outline, write the full outline now.
            - If the user asked for code, write the code now.
            - Ignore the [OFFER]/[INTERJECTION] format. Just write normally and professionally.
            - Ensure the final output is comprehensive and directly addresses the user's prompt.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ==============================================================================
# 4. STATE DEFINITIONS
# ==============================================================================

DebateAction = Literal["CONTINUE_DEBATE", "PRESENT_FINDINGS"]

class ProcessStep(TypedDict):
    id: str
    step: str
    messages: Annotated[list[BaseMessage], operator.add]
    category: Literal["lead", "appointment", "contribution", "human"]

class CollaborativeState(MessagesState):
    lead_agent: Annotated[list[str], operator.add]
    human_inputs: Annotated[list[str], operator.add]
    steps: Annotated[list[ProcessStep], operator.add]

def reduce_fanouts(left, right):
    if left is None:
        left = []
    if not right:
        return []
    return left + right

class ContributorInputState(MessagesState):
    consolidation_id: str
    agent_name: str

class ContributorOutputState(TypedDict):
    contributions: Annotated[list[BaseMessage], reduce_fanouts]

# ==============================================================================
# 5. HELPER FUNCTIONS
# ==============================================================================

def format_participants(participants: {}, exclude: list[str] = []) -> str:
    return "\n".join(
        [
            f"display_name: {info['display_name']}\nprofile: {info['profile']}"
            for name, info in participants.items()
            if name not in exclude
        ]
    )

def format_contributions(contributions: list[BaseMessage]) -> str:
    contributions_string = "\n\n".join([f"{message.name}: {message.content}" for message in contributions])
    return """
Below are the opinions of the other participants. Take them into an account only if they are relevant to your opinion and if you want to build on top of them.
========================================
{contributions_string}
========================================
""".format(contributions_string=contributions_string)

def get_step_messages(state: CollaborativeState, lead_agent_name: str, agent_contributions_last: int, listen_last: int) -> list[BaseMessage]:
    messages = []
    for step in reversed(state["steps"]):
        if step["category"] in ["human", "lead", "appointment"]:
            step_messages = []
            for message in reversed(step["messages"]):
                if message.type == "ai" and message.name != lead_agent_name:
                    content = f"Response from\n-----------------------------------\n{message.name}: {message.content}"
                    message = HumanMessage(content=content, name=message.name)
                step_messages.append(message)

            messages.extend(step_messages)
        elif step["category"] == "contribution":
            contributions = [
                message
                for message in step["messages"]
                if (agent_contributions_last > 0 and message.name == lead_agent_name) or listen_last > 0
            ]
            listen_last -= 1
            agent_contributions_last -= 1

            contributions = [message for message in contributions if message.content != "[PASS]"]
            message_ids = [message.id for message in messages]
            contributions = [message for message in contributions if message.id not in message_ids]

            if contributions:
                messages.append(HumanMessage(content=format_contributions(contributions)))

    return list(reversed(messages))

def get_agent_tools(agent_def):
    """Extracts actual tool functions based on YAML config."""
    if "tools" not in agent_def:
        return []
    # Map the string names from YAML to the functions in TOOL_MAP
    return [TOOL_MAP[t_name] for t_name in agent_def["tools"] if t_name in TOOL_MAP]

def sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Cleans messages for compatibility between Anthropic (tool_use blocks) 
    and Gemini (text-only expectation).
    """
    clean_messages = []
    for msg in messages:
        # Handle Anthropic's list-based content
        if isinstance(msg.content, list):
            text_content = ""
            for part in msg.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_content += part.get("text", "")
                    elif part.get("type") == "tool_use":
                        # Convert tool use to a text description for context so Gemini can see it happened
                        text_content += f"\n[Tool Call: {part.get('name')}]"
            
            # Create a new simplified message with string content
            if isinstance(msg, AIMessage):
                new_msg = AIMessage(content=text_content, name=msg.name if hasattr(msg, 'name') else 'assistant')
            elif isinstance(msg, HumanMessage):
                new_msg = HumanMessage(content=text_content, name=msg.name if hasattr(msg, 'name') else 'user')
            else:
                new_msg = AIMessage(content=str(text_content)) # Fallback
                
            clean_messages.append(new_msg)
        else:
            # Pass through standard string-based messages
            clean_messages.append(msg)
    return clean_messages

# ==============================================================================
# 6. NODE FUNCTIONS (With Tool Execution Logic)
# ==============================================================================

async def lead_agent_executor(state: CollaborativeState, config: RunnableConfig):
    # DYNAMIC LOAD: Get agents from config
    agents = get_agents_from_config(config)
    agent_names = list(agents.keys())
    all_participants = agents | {"human": HUMAN}

    # --- SAFETY CHECK FOR PANEL SWITCHING ---
    if not state.get("lead_agent"):
        # No history, pick default
        if not agent_names: raise ValueError("CRITICAL ERROR: No agents loaded in lead_agent_executor.")
        lead_agent_name = agent_names[0]
    else:
        # History exists, check validity
        current_history_lead = state["lead_agent"][-1]
        if current_history_lead in agents:
            lead_agent_name = current_history_lead
        else:
            print(f"--- INFO: Agent '{current_history_lead}' not found in current panel. Switching to '{agent_names[0]}'. ---")
            lead_agent_name = agent_names[0]
    # ----------------------------------------
        
    lead_agent_def = agents[lead_agent_name]

    await adispatch_custom_event(
        "lead_agent_executor",
        {"agent_name": lead_agent_name},
        config=config,
    )

    llm = LLMS[lead_agent_def["llm"] if "llm" in lead_agent_def else DEFAULT_LEAD_LLM]
    
    # BIND TOOLS IF AVAILABLE
    tools = get_agent_tools(lead_agent_def)
    if tools:
        llm = llm.bind_tools(tools)
    
    lead_agent = lead_agent_prompt | llm

    messages = get_step_messages(state, lead_agent_name, DEFAULT_LEAD_AGENT_CONTRIBUTIONS_LAST, lead_agent_def.get("lead_listen_last", DEFAULT_LEAD_LISTEN_LAST))
    
    # --- SANITIZE MESSAGES FOR GEMINI COMPATIBILITY ---
    messages = sanitize_messages(messages)

    chain_input = {
        "name": lead_agent_name,
        "display_name": lead_agent_def["display_name"],
        "persona": lead_agent_def["persona"],
        "role": lead_agent_def["role"],
        "messages": messages,
        "participants": format_participants(
            all_participants, exclude=[lead_agent_name]
        ),
    }

    # EXECUTION LOOP FOR TOOLS
    response = await lead_agent.ainvoke(chain_input, config=config)
    
    # Handle Tool Calls (Simple Loop - Max 3 Turns)
    max_turns = 3
    turn = 0
    while response.tool_calls and turn < max_turns:
        turn += 1
        # Create a temporary history list to feed back to the model
        temp_history = messages.copy() # Use the list derived from get_step_messages
        temp_history.append(response) # Add the AIMessage with tool_calls
        
        for tool_call in response.tool_calls:
            tool_func = TOOL_MAP.get(tool_call["name"])
            if tool_func:
                print(f"--- EXEC: {lead_agent_name} calling {tool_call['name']} ---")
                try:
                    tool_output = tool_func.invoke(tool_call["args"])
                except Exception as e:
                    tool_output = f"Error executing tool: {e}"
                
                # Create the Tool Result Message
                tool_msg = ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                temp_history.append(tool_msg)
        
        # Update messages for next iteration and re-invoke
        chain_input["messages"] = temp_history
        # Update our local messages variable to keep context growing for this node step
        messages = temp_history 
        response = await lead_agent.ainvoke(chain_input, config=config)

    response.name = lead_agent_name
    response.additional_kwargs["category"] = "lead"

    return {
        "messages": [response],
        "lead_agent": [lead_agent_name], # Update state with valid agent
        "steps": [
            {
                "id": str(uuid.uuid4()),
                "step": "lead_agent_executor",
                "messages": [response],
                "category": "lead",
            }
        ],
    }

async def contributor_agent_executor(state: ContributorInputState, config: RunnableConfig) -> ContributorOutputState:
    # DYNAMIC LOAD
    agents = get_agents_from_config(config)
    all_participants = agents | {"human": HUMAN}

    contributor_agent_info = agents[state["agent_name"]]
    
    await adispatch_custom_event(
        "contributor_agent_executor",
        {"agent_name": state["agent_name"]},
        config=config,
    )
    llm = LLMS[
        (
            contributor_agent_info["llm"]
            if "llm" in contributor_agent_info
            else DEFAULT_CONTRIBUTOR_LLM
        )
    ]

    # BIND TOOLS IF AVAILABLE
    tools = get_agent_tools(contributor_agent_info)
    if tools:
        llm = llm.bind_tools(tools)

    contributor_agent = contributor_agent_prompt | llm
    
    chain_input = {
        "name": state["agent_name"],
        "display_name": contributor_agent_info["display_name"],
        "persona": contributor_agent_info["persona"],
        "role": contributor_agent_info["role"],
        "messages": state["messages"],
        "participants": format_participants(
            all_participants, exclude=[state["agent_name"]]
        ),
    }

    # EXECUTION LOOP FOR TOOLS
    response = await contributor_agent.ainvoke(chain_input, config=config)

    # Handle Tool Calls (Simple Loop - Max 3 Turns)
    max_turns = 3
    turn = 0
    while response.tool_calls and turn < max_turns:
        turn += 1
        # Create a temporary history list to feed back to the model
        temp_history = state["messages"].copy()
        temp_history.append(response) # Add the AIMessage with tool_calls
        
        for tool_call in response.tool_calls:
            tool_func = TOOL_MAP.get(tool_call["name"])
            if tool_func:
                print(f"--- EXEC: {state['agent_name']} calling {tool_call['name']} ---")
                try:
                    tool_output = tool_func.invoke(tool_call["args"])
                except Exception as e:
                    tool_output = f"Error executing tool: {e}"
                
                # Create the Tool Result Message
                tool_msg = ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                temp_history.append(tool_msg)
        
        # Update input for next iteration
        chain_input["messages"] = temp_history
        response = await contributor_agent.ainvoke(chain_input, config=config)

    response.name = state["agent_name"]
    response.additional_kwargs["consolidation_id"] = state["consolidation_id"]
    response.additional_kwargs["category"] = "contributor"

    return {"contributions": [response], "messages": [response]}

async def debate_director_node(state: CollaborativeState, config: RunnableConfig) -> dict:
    # DYNAMIC LOAD
    agents = get_agents_from_config(config)
    agent_names = list(agents.keys())

    # --- SAFETY CHECK FOR PANEL SWITCHING ---
    if not state.get("lead_agent"):
         if not agent_names: raise ValueError("CRITICAL ERROR: No agents loaded in debate_director.")
         lead_agent_name = agent_names[0]
    else:
        current_history_lead = state["lead_agent"][-1]
        if current_history_lead in agents:
            lead_agent_name = current_history_lead
        else:
            lead_agent_name = agent_names[0]
    # ----------------------------------------
        
    lead_agent_def = agents[lead_agent_name]

    await adispatch_custom_event(
        "debate_director_node",
        {"agent_name": lead_agent_name},
        config=config,
    )

    llm = LLMS[lead_agent_def["llm"] if "llm" in lead_agent_def else DEFAULT_LEAD_LLM]
    director_agent = debate_director_prompt | llm

    messages = get_step_messages(state, lead_agent_name, DEFAULT_LEAD_AGENT_CONTRIBUTIONS_LAST, 1)

    # --- SANITIZE MESSAGES FOR GEMINI COMPATIBILITY ---
    messages = sanitize_messages(messages)

    # --- ADDED config=config FOR STREAMING ---
    response = await director_agent.ainvoke({"messages": messages, "name": lead_agent_name}, config=config)
    
    decision = response.content.strip().upper()
    
    if "CONTINUE_DEBATE" in decision:
        parsed_decision = "CONTINUE_DEBATE"
    elif "PRESENT_FINDINGS" in decision:
        parsed_decision = "PRESENT_FINDINGS"
    else:
        parsed_decision = "PRESENT_FINDINGS"

    update_step = {
        "id": str(uuid.uuid4()),
        "step": "debate_director_node",
        "messages": [AIMessage(content=f"Decision: {parsed_decision}", name=lead_agent_name)],
        "category": "lead",
        "decision": parsed_decision
    }
    
    decision_message = AIMessage(content=parsed_decision, name="debate_director")
    
    return {
        "steps": [update_step],
        "messages": [decision_message]
    }

async def final_response_node(state: CollaborativeState, config: RunnableConfig):
    """
    Executes the Lead Agent one last time to synthesize the findings into a proper response.
    """
    print("--- EXEC: Final Synthesis ---")
    agents = get_agents_from_config(config)
    
    # Identify Lead Agent (Safety Check included)
    if state.get("lead_agent") and state["lead_agent"][-1] in agents:
        lead_agent_name = state["lead_agent"][-1]
    else:
        lead_agent_name = list(agents.keys())[0]

    lead_agent_def = agents[lead_agent_name]
    
    # --- CRITICAL FIX: DISPATCH EVENT SO UI KNOWS TO CREATE A MESSAGE BOX ---
    await adispatch_custom_event(
        "final_response_node", 
        {"agent_name": lead_agent_name}, 
        config=config
    )

    # Use the Final Response Prompt
    llm = LLMS[lead_agent_def.get("llm", DEFAULT_LEAD_LLM)]
    
    # Bind tools here too, in case final synthesis needs one last check
    tools = get_agent_tools(lead_agent_def)
    if tools:
        llm = llm.bind_tools(tools)

    chain = final_response_prompt | llm
    
    # --- SANITIZE MESSAGES FOR GEMINI COMPATIBILITY ---
    clean_messages = sanitize_messages(state["messages"])

    # --- ADDED config=config FOR STREAMING ---
    response = await chain.ainvoke({
        "display_name": lead_agent_def["display_name"],
        "messages": clean_messages  # Use sanitized history
    }, config=config)
    
    response.name = lead_agent_name
    
    return {
        "messages": [response],
        "steps": [{"id": str(uuid.uuid4()), "step": "final_response_node", "messages": [response], "category": "lead"}]
    }

def appoint_lead_agent(state: CollaborativeState, agent_name: str, human_input: str):
    contribution_step = next((step for step in reversed(state["steps"]) if step["category"] == "contribution"), None)
    agent_contribution = (
        next(
            (
                message
                for message in contribution_step["messages"]
                if message.name == agent_name
                and message.content != "[PASS]"
            ),
            None,
        )
    )
    messages: list[BaseMessage] = []
    if agent_contribution:
        messages.append(
            AIMessage(
                content=agent_contribution.content, name=agent_name
            )
        )
    messages.append(HumanMessage(content=human_input))
    return {
        "lead_agent": [agent_name],
        "messages": messages,
        "steps": [
            {
                "id": str(uuid.uuid4()),
                "step": "appoint_lead_agent",
                "messages": messages,
                "category": "appointment",
            }
        ],
    }

def human_input_received_node(state: CollaborativeState, config: RunnableConfig):
    # DYNAMIC LOAD
    agents = get_agents_from_config(config)
    agent_names = list(agents.keys())
    
    # SAFETY: If agents are empty, we cannot proceed.
    if not agent_names:
        raise ValueError("CRITICAL ERROR: No agents loaded from panel configuration.")

    human_input = state["human_inputs"][-1]
    agent_match = re.search(r"@(\w+)", human_input)
    
    if agent_match:
        agent_name = agent_match.group(1)
        if agent_name in agents:
            return appoint_lead_agent(state, agent_name, human_input)
    
    current_lead_agents = state.get("lead_agent")
    
    # --- SAFETY CHECK FOR PANEL SWITCHING ---
    if not current_lead_agents:
        lead_agent_to_set = [agent_names[0]] 
    else:
        # Check if history matches current panel
        last_agent = current_lead_agents[-1]
        if last_agent in agents:
            lead_agent_to_set = [last_agent]
        else:
            # Panel switched, default to new panel's leader
            lead_agent_to_set = [agent_names[0]]
    # ----------------------------------------

    return {
        "lead_agent": lead_agent_to_set,
        "messages": [HumanMessage(content=human_input)],
        "steps": [
            {
                "id": str(uuid.uuid4()),
                "step": "human_input_received_node",
                "messages": [HumanMessage(content=human_input)],
                "category": "human",
            }
        ],
    }

def consolidate_contributions_node(state: ContributorOutputState):
    consolidation_id = state["contributions"][-1].additional_kwargs["consolidation_id"]
    consolidated_messages = [
        message
        for message in state["contributions"]
    ]
    return {
        "steps": [
            {
                "id": consolidation_id,
                "step": "consolidate_contributions_node",
                "messages": consolidated_messages,
                "category": "contribution",
            }
        ],
        "contributions": None,
    }

# ==============================================================================
# 7. EDGE FUNCTIONS
# ==============================================================================

def debate_director_router(state: CollaborativeState) -> DebateAction:
    last_message = state["messages"][-1].content
    if last_message == "CONTINUE_DEBATE":
        return "CONTINUE_DEBATE"
    return "PRESENT_FINDINGS"

def human_input_decision_edge(state: CollaborativeState) -> Literal["END", "LEAD"]:
    message = state["messages"][-1].content
    if message == "[END]":
        return "END"
    return "LEAD"

def create_contributor_executors_edge(state: CollaborativeState, config: RunnableConfig):
    # DYNAMIC LOAD
    agents = get_agents_from_config(config)
    
    # SAFETY: Stop execution if agents dictionary is empty
    if not agents:
        print("--- ERROR: Agents dictionary is empty. Cannot start contributors. ---")
        return []
    
    consolidation_id = str(uuid.uuid4())
    
    # --- SAFETY CHECK FOR PANEL SWITCHING ---
    if state["lead_agent"] and state["lead_agent"][-1] in agents:
        lead_agent_name = state["lead_agent"][-1] 
    else:
        lead_agent_name = list(agents.keys())[0]
    # ----------------------------------------
    
    # Filter contributors
    contributors = {name: d for name, d in agents.items() if name != lead_agent_name}
    
    sends = []
    for agent_name, agent_def in contributors.items():
        sends.append(
            Send(
                "contributor_agent_executor",
                {
                    "consolidation_id": consolidation_id,
                    "agent_name": agent_name,
                    "messages": get_step_messages(state, agent_name, DEFAULT_CONTRIBUTOR_AGENT_CONTRIBUTIONS_LAST, agent_def.get("contributor_listen_last", DEFAULT_CONTRIBUTOR_LISTEN_LAST)),
                },
            )
        )
        time.sleep(0.5) 
        
    return sends

# ==============================================================================
# 8. GRAPH CONSTRUCTION
# ==============================================================================

def create_graph() -> Tuple[StateGraph, CompiledStateGraph]:
    memory = MemorySaver()

    workflow = StateGraph(CollaborativeState)

    workflow.add_node("human_input_received_node", human_input_received_node)
    workflow.add_node("lead_agent_executor", lead_agent_executor)
    workflow.add_node("debate_director_node", debate_director_node)
    workflow.add_node("consolidate_contributions_node", consolidate_contributions_node)
    workflow.add_node("contributor_agent_executor", contributor_agent_executor)
    
    # NEW NODE: Final Response
    workflow.add_node("final_response_node", final_response_node)

    workflow.set_entry_point("human_input_received_node")
    workflow.add_conditional_edges(
        "human_input_received_node",
        human_input_decision_edge,
        {"END": END, "LEAD": "lead_agent_executor"},
    )

    workflow.add_conditional_edges(
        "lead_agent_executor", create_contributor_executors_edge, ["contributor_agent_executor"]
    )
    workflow.add_edge("contributor_agent_executor", "consolidate_contributions_node")
    
    workflow.add_edge("consolidate_contributions_node", "debate_director_node")
    
    # ROUTING UPDATE: Present Findings -> Final Response Node
    workflow.add_conditional_edges(
        "debate_director_node",
        debate_director_router, 
        {
            "CONTINUE_DEBATE": "lead_agent_executor", 
            "PRESENT_FINDINGS": "final_response_node", 
        }
    )
    
    workflow.add_edge("final_response_node", END)

    graph = workflow.compile(checkpointer=memory)

    return workflow, graph

if __name__ == "__main__":
    # Test execution
    workflow, graph = create_graph()
    print("Graph compiled successfully.")