import operator
import re
import time
from typing import Annotated, Literal, Tuple, TypedDict

# --- FIX: Updated Imports to use langchain_core and correct paths ---
# The original error was: ModuleNotFoundError: No module named 'langchain.schema.runnables' (now in langchain_core)
# The second error was: ModuleNotFoundError: No module named 'langchain.prompts' (now in langchain_core.prompts)
# The third error was: ModuleNotFoundError: No module named 'langchain.schema' (now in langchain_core)

from agents.loader import load_human, load_panel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # FIXED
from langchain_core.runnables.config import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks.manager import adispatch_custom_event # FIXED
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage # FIXED
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
import uuid

load_dotenv()

# --- LLM INITIALIZATION WITH ENHANCED RATE LIMIT HANDLING ---
# Setting max_retries higher increases the chance of recovering from temporary 429 errors.
openai_llm = ChatOpenAI(model="gpt-4o", temperature=0.7, max_retries=5)
openai_llm_mini = ChatOpenAI(model="gpt-4o-mini", max_retries=5)

# Anthropic (Increased retries)
anthropic_llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", max_retries=5)

# Gemini (Increased retries for better 429 handling)
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_retries=5)

### CONFIG START ###

LLMS = {
    "openai": openai_llm,
    "openai_mini": openai_llm_mini,
    "anthropic": anthropic_llm,
    "gemini": gemini_llm,
}

# default LLM for the lead agent
DEFAULT_LEAD_LLM = "gemini" # "openai"
# default LLM for the contributor agents
DEFAULT_CONTRIBUTOR_LLM = "anthropic" # "openai"

# panel to use - see agents/panels/
AGENTS_SET = "creative_panel"

# how many contributions from the same agent to take into account
DEFAULT_LEAD_AGENT_CONTRIBUTIONS_LAST = 10
DEFAULT_CONTRIBUTOR_AGENT_CONTRIBUTIONS_LAST = 5

# how many contributions from other agents to take into account
DEFAULT_LEAD_LISTEN_LAST = 5
DEFAULT_CONTRIBUTOR_LISTEN_LAST = 2

### CONFIG END ###

# Ensure AGENTS is loaded; assuming load_panel and AGENTS is correctly defined
try:
    AGENTS = load_panel(AGENTS_SET)
    HUMAN = load_human()
    ALL_PARTICIPANTS = AGENTS | {"human": HUMAN}
    # Get a list of agent names for initial lead agent selection
    AGENT_NAMES = list(AGENTS.keys())
except Exception as e:
    print(f"Error loading agents: {e}")
    # Fallback to prevent immediate crash if agents cannot be loaded
    AGENTS = {}
    AGENT_NAMES = []
    ALL_PARTICIPANTS = {}


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

# New prompt for the internal debate director
debate_director_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """You are the Debate Director. Your current task is to review the most recent contributions from the panel members.
          
          Based on the contributions and the existing conversation history, decide the next step:
          1. **CONTINUE_DEBATE**: If there is significant disagreement (look for [CONTRAST] or [DISAGREE] tags) or if the topic is complex and requires further refinement/synthesis by the lead agent.
          2. **PRESENT_FINDINGS**: If the contributions are consistent, consensus is reached, or the ideas are sufficiently refined to be presented to the @human.

          Output **only** the selected keyword: CONTINUE_DEBATE or PRESENT_FINDINGS. Do not add any other text, explanation, or markdown formatting.
          """),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# New type for the director's decision
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
        # Overwrite
        return []
    return left + right


class ContributorInputState(MessagesState):
    consolidation_id: str
    agent_name: str


class ContributorOutputState(TypedDict):
    contributions: Annotated[list[BaseMessage], reduce_fanouts]


def format_participants(participants: {}, exclude: list[str] = []) -> str:
    # Syntax fix applied: using .format() instead of f-string literal for multiline string
    return "\n".join(
        [
            """
------------------------------
display_name: {display_name}
profile: {profile}
""".format(
                display_name=participant_info['display_name'], 
                profile=participant_info['profile']
            )
            for participant, participant_info in participants.items()
            if participant not in exclude
        ]
    )


def format_contributions(contributions: list[BaseMessage]) -> str:
    # 1. Generate the dynamic content string first
    contributions_string = "\n\n".join([f"{message.name}: {message.content}" for message in contributions])
    
    # 2. Use a standard triple-quoted string template with .format()
    return """
Below are the opinions of the other participants. Take them into an account only if they are relevant to your opinion and if you want to build on top of them.
========================================
{contributions_string}
========================================
""".format(contributions_string=contributions_string)


def get_step_messages(state: CollaborativeState, lead_agent_name: str, agent_contributions_last: int, listen_last: int) -> list[BaseMessage]:
    # walk through the steps backwards. Collect all human, lead, appointment messages. Collect the contributions from the last contribution step and all the contributions of this agent
    messages = []
    for step in reversed(state["steps"]):
        if step["category"] in ["human", "lead", "appointment"]:
            # the AI messages with other agents should be converted to human messages
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

            # clean up the contributions that are [PASS]
            contributions = [message for message in contributions if message.content != "[PASS]"]

            # cleanup messages with id already in messages
            message_ids = [message.id for message in messages]
            contributions = [message for message in contributions if message.id not in message_ids]

            if contributions:
                messages.append(HumanMessage(content=format_contributions(contributions)))

    return list(reversed(messages))


async def lead_agent_executor(
    state: CollaborativeState, config: RunnableConfig
):
    """
    Execute the lead agent's response based on the current state.
    """
    # FIX: Safety check for initial run (although addressed in human_input_received_node)
    if not state.get("lead_agent"):
        if not AGENT_NAMES:
            raise ValueError("No agents defined to act as the lead agent.")
        lead_agent_name = AGENT_NAMES[0] # Fallback to the first defined agent
    else:
        # Original logic: get the last lead agent name
        lead_agent_name = state["lead_agent"][-1]
        
    lead_agent_def = AGENTS[lead_agent_name]

    await adispatch_custom_event(
        "lead_agent_executor",
        {"agent_name": lead_agent_name},
        config=config,
    )

    llm = LLMS[lead_agent_def["llm"] if "llm" in lead_agent_def else DEFAULT_LEAD_LLM]
    lead_agent = lead_agent_prompt | llm

    messages = get_step_messages(state, lead_agent_name, DEFAULT_LEAD_AGENT_CONTRIBUTIONS_LAST, lead_agent_def.get("lead_listen_last", DEFAULT_LEAD_LISTEN_LAST))
    
    response = await lead_agent.ainvoke(
        {
            "name": lead_agent_name,
            "display_name": lead_agent_def["display_name"],
            "persona": lead_agent_def["persona"],
            "role": lead_agent_def["role"],
            "messages": messages,
            "participants": format_participants(
                ALL_PARTICIPANTS, exclude=[lead_agent_name]
            ),
        }
    )
    response.name = lead_agent_name
    response.additional_kwargs["category"] = "lead"

    return {
        "messages": [response],
        "steps": [
            {
                "id": str(uuid.uuid4()),
                "step": "lead_agent_executor",
                "messages": [response],
                "category": "lead",
            }
        ],
    }


async def contributor_agent_executor(
    state: ContributorInputState, config: RunnableConfig
) -> ContributorOutputState:
    contributor_agent_info = AGENTS[state["agent_name"]]
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
    contributor_agent = contributor_agent_prompt | llm
    result = await contributor_agent.ainvoke(
        {
            "name": state["agent_name"],
            "display_name": contributor_agent_info["display_name"],
            "persona": contributor_agent_info["persona"],
            "role": contributor_agent_info["role"],
            "messages": state["messages"],
            "participants": format_participants(
                ALL_PARTICIPANTS, exclude=[state["agent_name"]]
            ),
        }
    )
    result.name = state["agent_name"]
    result.additional_kwargs["consolidation_id"] = state["consolidation_id"]
    result.additional_kwargs["category"] = "contributor"

    return {"contributions": [result], "messages": [result]}


async def debate_director_node(state: CollaborativeState, config: RunnableConfig) -> dict:
    """
    Decides if the internal debate should continue based on the latest contributions and updates the state with the decision.
    """
    # Safety check for initial run
    if not state.get("lead_agent"):
         if not AGENT_NAMES: raise ValueError("No agents defined.")
         lead_agent_name = AGENT_NAMES[0]
    else:
        lead_agent_name = state["lead_agent"][-1]
        
    lead_agent_def = AGENTS[lead_agent_name]

    await adispatch_custom_event(
        "debate_director_node",
        {"agent_name": lead_agent_name},
        config=config,
    )

    llm = LLMS[lead_agent_def["llm"] if "llm" in lead_agent_def else DEFAULT_LEAD_LLM]
    director_agent = debate_director_prompt | llm

    # The messages should include the consolidated contributions for the decision
    messages = get_step_messages(state, lead_agent_name, DEFAULT_LEAD_AGENT_CONTRIBUTIONS_LAST, 1)

    response = await director_agent.ainvoke({"messages": messages, "name": lead_agent_name})
    
    # Strip any potential markdown/whitespace and extract the decision
    decision = response.content.strip().upper()
    
    # Simple keyword extraction for robust routing
    if "CONTINUE_DEBATE" in decision:
        parsed_decision = "CONTINUE_DEBATE"
    elif "PRESENT_FINDINGS" in decision:
        parsed_decision = "PRESENT_FINDINGS"
    else:
        # Fallback if the LLM output is garbled
        parsed_decision = "PRESENT_FINDINGS"

    # Store the decision in the state so the conditional edge function can read it.
    update_step = {
        "id": str(uuid.uuid4()),
        "step": "debate_director_node",
        "messages": [AIMessage(content=f"Decision: {parsed_decision}", name=lead_agent_name)],
        "category": "lead",
        "decision": parsed_decision
    }
    
    # Create a dummy message object to store the decision in the main message history
    decision_message = AIMessage(content=parsed_decision, name="debate_director")
    
    return {
        "steps": [update_step],
        "messages": [decision_message]
    }

def debate_director_router(state: CollaborativeState) -> DebateAction:
    """
    Reads the decision from the last step's message content and returns the routing key.
    """
    last_message = state["messages"][-1].content
    
    if last_message == "CONTINUE_DEBATE":
        return "CONTINUE_DEBATE"
    
    return "PRESENT_FINDINGS"


def format_conversation_state(state: CollaborativeState) -> str:
    prev_message_str = state["messages"][-1].content if state["messages"] else ""
    contribution_step = next((step for step in reversed(state["steps"]) if step["category"] == "contribution"), None)
    agent_contributions_str = ""
    if contribution_step:
        agent_contributions = [
            message
            for message in contribution_step["messages"]
            if message.content != "[PASS]"
        ]
        agent_contributions_str = "\n\n".join(
            [f"{message.name}: {message.content}" for message in agent_contributions]
        )
    return f"@{state['lead_agent'][-1]}: {prev_message_str}\n\n{agent_contributions_str}\n\nHuman: "

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

# --- FIX IMPLEMENTED HERE (Resolves the initial IndexError: list index out of range) ---
def human_input_received_node(state: CollaborativeState):
    """
    Handles new human input, checking for an explicit lead agent appointment (@agent_name)
    and ensuring the 'lead_agent' state is initialized for the first turn.
    """
    human_input = state["human_inputs"][-1]
    agent_match = re.search(r"@(\w+)", human_input)
    
    # 1. Check for explicit agent appointment
    if agent_match:
        agent_name = agent_match.group(1)
        if agent_name in AGENTS:
            return appoint_lead_agent(state, agent_name, human_input)
    
    # 2. Handle the default case (no explicit appointment)
    
    # Determine the lead agent for the next step.
    # If lead_agent is not set (first run), use the first agent from AGENT_NAMES as default.
    # If it is set, keep the last appointed agent.
    current_lead_agents = state.get("lead_agent")
    if not current_lead_agents:
        if not AGENT_NAMES:
             # Should not happen if AGENTS are loaded correctly, but good to check.
             raise ValueError("No agents defined to set as default lead agent.")
        lead_agent_to_set = [AGENT_NAMES[0]] # Set default lead agent for the first run
    else:
        # Keep the existing lead agent
        lead_agent_to_set = [current_lead_agents[-1]] 

    # Return the state update, ensuring 'lead_agent' is populated
    return {
        "lead_agent": lead_agent_to_set, # <-- CRITICAL FIX
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
# --- END FIX ---


def human_input_decision_edge(
    state: CollaborativeState,
) -> Literal["END", "LEAD"]:
    message = state["messages"][-1].content
    if message == "[END]":
        return "END"

    return "LEAD"


def create_contributor_executors_edge(state: CollaborativeState):
    # all the agents except the lead one
    consolidation_id = str(uuid.uuid4())
    # filter out the lead agent and still keep the dict
    lead_agent_name = state["lead_agent"][-1] # This is safe now due to the fix
    agents = {agent_name: agent_def for agent_name, agent_def in AGENTS.items() if agent_name != lead_agent_name}
    
    sends = []
    # Loop over agents and apply a synchronous sleep delay for rate limit management
    for agent_name, agent_def in agents.items():
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
        # Synchronous sleep to throttle the concurrent API calls for rate limit compliance (e.g., free tier 10 RPM)
        time.sleep(2) 
        
    return sends


def consolidate_contributions_node(state: ContributorOutputState):
    # filer the contributions by the consolidation_id
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


def create_graph() -> Tuple[StateGraph, CompiledStateGraph]:
    memory = MemorySaver()

    # Create the graph
    workflow = StateGraph(CollaborativeState)

    # Add nodes
    workflow.add_node("human_input_received_node", human_input_received_node)
    workflow.add_node("lead_agent_executor", lead_agent_executor)
    workflow.add_node("debate_director_node", debate_director_node)
    workflow.add_node("consolidate_contributions_node", consolidate_contributions_node)

    workflow.add_node("contributor_agent_executor", contributor_agent_executor)

    # Add edges
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
    
    # 1. Edge from consolidation to the director node (which now returns a state dict)
    workflow.add_edge("consolidate_contributions_node", "debate_director_node")
    
    # 2. Conditional edge from the director node, using the router function
    workflow.add_conditional_edges(
        "debate_director_node",
        debate_director_router, # Use the router function to read the state
        {
            "CONTINUE_DEBATE": "lead_agent_executor", # Loop back to Lead for synthesis/refinement
            "PRESENT_FINDINGS": "human_input_received_node", # Present final response and wait for human input
        }
    )

    # Compile the graph
    graph = workflow.compile(
        checkpointer=memory, interrupt_before=["human_input_received_node"]
    )

    return workflow, graph