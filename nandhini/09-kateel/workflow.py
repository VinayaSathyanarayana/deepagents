import operator
import re
import time
import os
import yaml
import uuid
import functools
import json
import sys
from typing import Annotated, Literal, Tuple, TypedDict, Dict, Any, List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

# ==============================================================================
# 🛠️ USER CONFIGURATION (EDIT SETTINGS HERE)
# ==============================================================================

# --- MODEL SELECTION ---
# Primary Models
CONF_MODEL_OPENAI = "gpt-4o"
CONF_MODEL_OPENAI_MINI = "gpt-4o-mini"
CONF_MODEL_ANTHROPIC = "claude-3-haiku-20240307"

# Gemini Fallback List (Tries these in order)
CONF_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash"
]

# --- AGENT BEHAVIOR ---
CONF_DEFAULT_LEAD_LLM = "gemini"        # 'gemini', 'openai', 'anthropic'
CONF_DEFAULT_CONTRIBUTOR_LLM = "anthropic"
CONF_LLM_MAX_RETRIES = 5                # How many times to retry API calls on failure

# --- CONTEXT MEMORY (How much history agents see) ---
# Reduce these numbers if you hit Token Limits
CONF_LEAD_READ_DEPTH = 15               # How far back the Lead Agent reads
CONF_CONTRIBUTOR_READ_DEPTH = 8         # How far back Contributors read
CONF_LEAD_LISTEN_DEPTH = 5              # Recent turns Lead Agent pays close attention to
CONF_CONTRIBUTOR_LISTEN_DEPTH = 2       # Recent turns Contributors pay close attention to

# ==============================================================================
# 0. SAFETY & IMPORTS
# ==============================================================================
print("--- INFO: Initializing workflow.py ---")

# Fix for Streamlit Conflict
if "streamlit" not in sys.modules:
    import types
    sys.modules["streamlit"] = types.ModuleType("streamlit")

# Safe Import of Tools
try:
    from financial_tools import TOOL_MAP
    print("--- INFO: Loaded financial_tools successfully ---")
except Exception as e:
    print(f"--- WARNING: Could not import financial_tools ({e}). Proceeding without tools. ---")
    TOOL_MAP = {}

# ==============================================================================
# 1. DYNAMIC LOADER LOGIC
# ==============================================================================

def load_human():
    return {
        "display_name": "User",
        "persona": "You are the user interacting with the panel.",
        "role": "User",
        "profile": "The human user seeking assistance."
    }

@functools.lru_cache(maxsize=10)
def load_panel_cached(panel_name: str):
    file_path = os.path.join("agents", "panels", f"{panel_name}.yaml")
    if not os.path.exists(file_path): return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except: return {}

    agents = {}
    ignored_keys = ['scoring_rubric', 'workflow', 'deliverables_pack', 'governance', 'purpose', 'version']
    if data:
        for key, value in data.items():
            if key in ignored_keys: continue
            if isinstance(value, dict):
                if 'persona' in value: agents[key] = value
                else:
                    for sk, sv in value.items():
                        if isinstance(sv, dict) and 'persona' in sv: agents[sk] = sv
    return agents

def get_agents_from_config(config: RunnableConfig) -> Dict[str, Any]:
    panel_name = config.get("configurable", {}).get("panel_name", "abstract_evaluation_panel")
    agents = load_panel_cached(panel_name)
    if not agents: agents = load_panel_cached("abstract_evaluation_panel")
    return agents

# ==============================================================================
# 2. LLM INITIALIZATION
# ==============================================================================

try:
    openai_llm = ChatOpenAI(model=CONF_MODEL_OPENAI, temperature=0.7, max_retries=CONF_LLM_MAX_RETRIES)
    openai_llm_mini = ChatOpenAI(model=CONF_MODEL_OPENAI_MINI, max_retries=CONF_LLM_MAX_RETRIES)
    anthropic_llm = ChatAnthropic(model=CONF_MODEL_ANTHROPIC, max_retries=CONF_LLM_MAX_RETRIES)
except:
    openai_llm = None
    anthropic_llm = None

# Robust Gemini Initialization
gemini_llm = None
for model_name in CONF_GEMINI_MODELS:
    try:
        # Try initializing with a dummy call or just object creation
        gemini_llm = ChatGoogleGenerativeAI(model=model_name, max_retries=CONF_LLM_MAX_RETRIES)
        print(f"--- INFO: Successfully loaded Gemini model: {model_name} ---")
        break
    except Exception:
        continue

if not gemini_llm:
    print("--- WARNING: Failed to load any Gemini models. ---")

LLMS = {
    "openai": openai_llm,
    "openai_mini": openai_llm_mini,
    "anthropic": anthropic_llm,
    "gemini": gemini_llm, 
}

HUMAN = load_human()

# ==============================================================================
# 3. PROMPTS
# ==============================================================================

contributor_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an actor playing: {display_name}
             Persona: {persona}
             Role: {role}
             Contribute ONLY if highly relevant. Be brief.
             Formats: [INTERJECTION], [OFFER], [PASS].
             Tags: [AGREE], [DISAGREE], [SUPPORT], [CLARIFY], [CONTRAST] + @display_name.
             Participants: {participants}"""),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

lead_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Lead Agent: {display_name}
             Persona: {persona}
             Role: {role}
             1. Engage @human concisely.
             2. Manage panel (@display_name).
             3. Refine consensus.
             Participants: {participants}"""),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

debate_director_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Debate Director.
            Decide next step: CONTINUE_DEBATE or PRESENT_FINDINGS.
            Output ONLY the keyword."""),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Strategic Planner.
            Context: {context_summary}
            
            Determine if the user's request requires a multi-step execution plan.
            
            OUTPUT STRICT JSON ONLY. No markdown.
            Format: {{ "rationale": "...", "steps": ["Step 1", "Step 2"] }}
            If simple, return: {{ "steps": [] }}
            """),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

final_response_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are the Lead Agent: {display_name}.
            
            **Task Results (Executed Plan):**
            {task_results}
            
            Synthesize all information (History + Task Results) and write the final response to the @human.
            - Provide a concrete, detailed answer.
            - Ignore [OFFER] tags.
            """),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ==============================================================================
# 4. STATE
# ==============================================================================

DebateAction = Literal["CONTINUE_DEBATE", "PRESENT_FINDINGS"]

class ProcessStep(TypedDict):
    id: str
    step: str
    messages: Annotated[list[BaseMessage], operator.add]
    category: Literal["lead", "appointment", "contribution", "human", "planning"]

class CollaborativeState(MessagesState):
    lead_agent: Annotated[list[str], operator.add]
    human_inputs: Annotated[list[str], operator.add]
    steps: Annotated[list[ProcessStep], operator.add]
    plan: Optional[List[str]]
    current_step_index: Optional[int]
    task_results: Annotated[List[str], operator.add]

def reduce_fanouts(left, right):
    if left is None: left = []
    if not right: return []
    return left + right

class ContributorInputState(MessagesState):
    consolidation_id: str
    agent_name: str

class ContributorOutputState(TypedDict):
    contributions: Annotated[list[BaseMessage], reduce_fanouts]

# ==============================================================================
# 5. HELPERS
# ==============================================================================

def format_participants(participants: {}, exclude: list[str] = []) -> str:
    return "\n".join([f"display_name: {info.get('display_name', name)}\nprofile: {info.get('profile', '')}" for name, info in participants.items() if name not in exclude])

def format_contributions(contributions: list[BaseMessage]) -> str:
    s = "\n\n".join([f"{m.name}: {m.content}" for m in contributions])
    return f"\nOther participants' opinions:\n================\n{s}\n================\n"

def get_step_messages(state: CollaborativeState, lead_agent_name: str, agent_contributions_last: int, listen_last: int) -> list[BaseMessage]:
    messages = []
    for step in reversed(state["steps"]):
        if step["category"] in ["human", "lead", "appointment", "planning"]:
            for message in reversed(step["messages"]):
                if message.type == "ai" and message.name != lead_agent_name:
                    content = f"Response from\n-----------------------------------\n{message.name}: {message.content}"
                    message = HumanMessage(content=content, name=message.name)
                messages.insert(0, message)
        elif step["category"] == "contribution":
            contributions = [m for m in step["messages"] if (agent_contributions_last > 0 and m.name == lead_agent_name) or listen_last > 0]
            listen_last -= 1
            agent_contributions_last -= 1
            contributions = [m for m in contributions if m.content != "[PASS]" and m.id not in [x.id for x in messages]]
            if contributions:
                messages.insert(0, HumanMessage(content=format_contributions(contributions)))
    return messages

def get_agent_tools(agent_def):
    if "tools" not in agent_def: return []
    return [TOOL_MAP[t_name] for t_name in agent_def["tools"] if t_name in TOOL_MAP]

def sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    clean_messages = []
    for msg in messages:
        if isinstance(msg.content, list):
            text_content = ""
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_content += part.get("text", "")
                elif isinstance(part, dict) and part.get("type") == "tool_use":
                    text_content += f"\n[Tool Call: {part.get('name')}]"
            
            if isinstance(msg, AIMessage): new_msg = AIMessage(content=text_content, name=msg.name if hasattr(msg, 'name') else 'assistant')
            elif isinstance(msg, HumanMessage): new_msg = HumanMessage(content=text_content, name=msg.name if hasattr(msg, 'name') else 'user')
            else: new_msg = AIMessage(content=str(text_content))
            clean_messages.append(new_msg)
        else:
            clean_messages.append(msg)
    return clean_messages

# ==============================================================================
# 6. NODES
# ==============================================================================

async def lead_agent_executor(state: CollaborativeState, config: RunnableConfig):
    agents = get_agents_from_config(config)
    lead_agent_name = state["lead_agent"][-1] if state.get("lead_agent") else list(agents.keys())[0]
    lead_agent_def = agents.get(lead_agent_name) or list(agents.values())[0]

    await adispatch_custom_event("lead_agent_executor", {"agent_name": lead_agent_name}, config=config)
    llm = LLMS[lead_agent_def.get("llm", CONF_DEFAULT_LEAD_LLM)]
    tools = get_agent_tools(lead_agent_def)
    if tools: llm = llm.bind_tools(tools)
    
    chain = lead_agent_prompt | llm
    
    # Use Configurable Listen Depth
    listen_depth = lead_agent_def.get("lead_listen_last", CONF_LEAD_LISTEN_DEPTH)
    messages = get_step_messages(state, lead_agent_name, CONF_LEAD_READ_DEPTH, listen_depth)
    messages = sanitize_messages(messages)

    response = await chain.ainvoke({
        "name": lead_agent_name,
        "display_name": lead_agent_def["display_name"],
        "persona": lead_agent_def["persona"],
        "role": lead_agent_def["role"],
        "messages": messages,
        "participants": format_participants(agents | {"human": HUMAN}, exclude=[lead_agent_name]),
    }, config=config)

    max_turns = 3
    turn = 0
    while response.tool_calls and turn < max_turns:
        turn += 1
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_func = TOOL_MAP.get(tool_call["name"])
            if tool_func:
                try: tool_output = tool_func.invoke(tool_call["args"])
                except Exception as e: tool_output = f"Error: {e}"
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"], name=tool_call["name"]))
        response = await chain.ainvoke({
            "name": lead_agent_name, "display_name": lead_agent_def["display_name"], "persona": lead_agent_def["persona"],
            "role": lead_agent_def["role"], "messages": messages,
            "participants": format_participants(agents | {"human": HUMAN}, exclude=[lead_agent_name]),
        }, config=config)

    response.name = lead_agent_name
    response.additional_kwargs["category"] = "lead"
    return {
        "messages": [response], "lead_agent": [lead_agent_name],
        "steps": [{"id": str(uuid.uuid4()), "step": "lead_agent_executor", "messages": [response], "category": "lead"}],
    }

async def contributor_agent_executor(state: ContributorInputState, config: RunnableConfig) -> ContributorOutputState:
    agents = get_agents_from_config(config)
    agent_info = agents[state["agent_name"]]
    
    await adispatch_custom_event("contributor_agent_executor", {"agent_name": state["agent_name"]}, config=config)
    llm = LLMS[agent_info.get("llm", CONF_DEFAULT_CONTRIBUTOR_LLM)]
    tools = get_agent_tools(agent_info)
    if tools: llm = llm.bind_tools(tools)
    
    chain = contributor_agent_prompt | llm
    chain_input = {
        "name": state["agent_name"], "display_name": agent_info["display_name"], "persona": agent_info["persona"],
        "role": agent_info["role"], "messages": state["messages"],
        "participants": format_participants(agents | {"human": HUMAN}, exclude=[state["agent_name"]]),
    }
    response = await chain.ainvoke(chain_input, config=config)
    
    max_turns = 3
    turn = 0
    while response.tool_calls and turn < max_turns:
        turn += 1
        state["messages"].append(response)
        for tool_call in response.tool_calls:
            tool_func = TOOL_MAP.get(tool_call["name"])
            if tool_func:
                try: tool_output = tool_func.invoke(tool_call["args"])
                except Exception as e: tool_output = f"Error: {e}"
                state["messages"].append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"], name=tool_call["name"]))
        chain_input["messages"] = state["messages"]
        response = await chain.ainvoke(chain_input, config=config)

    response.name = state["agent_name"]
    response.additional_kwargs["consolidation_id"] = state["consolidation_id"]
    return {"contributions": [response], "messages": [response]}

async def debate_director_node(state: CollaborativeState, config: RunnableConfig) -> dict:
    agents = get_agents_from_config(config)
    lead_name = state["lead_agent"][-1] if state.get("lead_agent") else list(agents.keys())[0]
    lead_def = agents.get(lead_name) or list(agents.values())[0]
    max_rounds = config.get("configurable", {}).get("max_debate_rounds", 2)
    
    consecutive_debates = 0
    for step in reversed(state["steps"]):
        if step["step"] == "debate_director_node": consecutive_debates += 1
        elif step["step"] == "planning_node": break
        elif step["step"] == "human_input_received_node": break
    
    if consecutive_debates >= max_rounds:
        parsed_decision = "PRESENT_FINDINGS"
    else:
        await adispatch_custom_event("debate_director_node", {"agent_name": lead_name}, config=config)
        llm = LLMS[lead_def.get("llm", CONF_DEFAULT_LEAD_LLM)]
        chain = debate_director_prompt | llm
        messages = sanitize_messages(get_step_messages(state, lead_name, CONF_LEAD_READ_DEPTH, 1))
        response = await chain.ainvoke({"messages": messages}, config=config)
        parsed_decision = "CONTINUE_DEBATE" if "CONTINUE_DEBATE" in response.content else "PRESENT_FINDINGS"

    msg = AIMessage(content=parsed_decision, name="debate_director")
    return {"steps": [{"id": str(uuid.uuid4()), "step": "debate_director_node", "messages": [msg], "category": "lead", "decision": parsed_decision}], "messages": [msg]}

async def planning_node(state: CollaborativeState, config: RunnableConfig) -> dict:
    print("--- EXEC: Planning Node ---")
    await adispatch_custom_event("planning_node", {"agent_name": "Planner"}, config=config)

    last_lead_msg = next((s["messages"][-1] for s in reversed(state["steps"]) if s["category"] == "lead"), None)
    task_results_update = []
    if state.get("current_step_index") is not None and last_lead_msg:
        task_results_update.append(f"Result of Step {state['current_step_index'] + 1}: {last_lead_msg.content}")

    current_plan = state.get("plan")
    current_index = state.get("current_step_index")

    if current_plan is None:
        llm = LLMS["gemini"]
        chain = planner_prompt | llm
        try:
            messages = sanitize_messages(state["messages"][-20:]) 
            response = await chain.ainvoke({"messages": messages, "context_summary": "Analyze user request for steps."}, config=config)
            
            content = response.content
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                json_str = match.group(0)
                plan_data = json.loads(json_str)
                current_plan = plan_data.get("steps", [])
            else:
                print(f"--- PLANNER WARNING: No JSON found ---")
                current_plan = []
            
            print(f"--- PLAN GENERATED: {len(current_plan)} steps ---")
            current_index = 0
            
        except Exception as e:
            print(f"--- PLANNER ERROR: {e} ---")
            current_plan = ["Analyze and answer the user's request comprehensively"]
            current_index = 0

    else:
        if current_index is not None: current_index += 1

    if current_plan and current_index is not None and current_index < len(current_plan):
        next_step = current_plan[current_index]
        print(f"--- EXEC STEP {current_index+1}: {next_step} ---")
        msg = HumanMessage(content=f"SYSTEM DIRECTIVE: Execute Step {current_index + 1}: '{next_step}'. Focus on this task.", name="Planner")
        return {
            "plan": current_plan, "current_step_index": current_index, "task_results": task_results_update,
            "messages": [msg], "steps": [{"id": str(uuid.uuid4()), "step": "planning_node", "messages": [msg], "category": "planning"}]
        }
    else:
        print("--- PLAN FINISHED ---")
        return {"current_step_index": None, "task_results": task_results_update}

async def final_response_node(state: CollaborativeState, config: RunnableConfig):
    agents = get_agents_from_config(config)
    final_agent = state["lead_agent"][-1] if state.get("lead_agent") else list(agents.keys())[0]
    agent_def = agents.get(final_agent) or list(agents.values())[0]
    
    await adispatch_custom_event("final_response_node", {"agent_name": final_agent}, config=config)
    llm = LLMS[agent_def.get("llm", CONF_DEFAULT_LEAD_LLM)]
    
    chain = final_response_prompt | llm
    task_results_str = "\n".join(state.get("task_results", []))
    
    if not task_results_str:
        task_results_str = "No structured plan executed. Refer to discussion history."

    response = await chain.ainvoke({"display_name": agent_def["display_name"], "messages": sanitize_messages(state["messages"]), "task_results": task_results_str}, config=config)
    response.name = final_agent
    return {"messages": [response], "steps": [{"id": str(uuid.uuid4()), "step": "final_response_node", "messages": [response], "category": "lead"}]}

def human_input_received_node(state: CollaborativeState, config: RunnableConfig):
    agents = get_agents_from_config(config)
    human_input = state["human_inputs"][-1]
    
    match = re.search(r"@(\w+)", human_input)
    if match and match.group(1) in agents:
        agent_name = match.group(1)
        return {
            "lead_agent": [agent_name], "messages": [HumanMessage(content=human_input)],
            "steps": [{"id": str(uuid.uuid4()), "step": "appoint_lead_agent", "messages": [HumanMessage(content=human_input)], "category": "appointment"}],
            "plan": None, "task_results": [], "current_step_index": None
        }

    current = state.get("lead_agent")
    lead = [current[-1]] if current and current[-1] in agents else [list(agents.keys())[0]]
    return {
        "lead_agent": lead, "messages": [HumanMessage(content=human_input)],
        "steps": [{"id": str(uuid.uuid4()), "step": "human_input_received_node", "messages": [HumanMessage(content=human_input)], "category": "human"}],
        "plan": None, "task_results": [], "current_step_index": None
    }

def consolidate_contributions_node(state: ContributorOutputState):
    return {"steps": [{"id": state["contributions"][-1].additional_kwargs["consolidation_id"], "step": "consolidate_contributions_node", "messages": list(state["contributions"]), "category": "contribution"}], "contributions": None}

# ==============================================================================
# 7. EDGES
# ==============================================================================

def create_contributor_executors_edge(state: CollaborativeState, config: RunnableConfig):
    agents = get_agents_from_config(config)
    lead = state["lead_agent"][-1] if state.get("lead_agent") else list(agents.keys())[0]
    contributors = {k: v for k, v in agents.items() if k != lead}
    cid = str(uuid.uuid4())
    sends = []
    for k, v in contributors.items():
        # Configurable listen depth
        listen_depth = v.get("contributor_listen_last", CONF_CONTRIBUTOR_LISTEN_DEPTH)
        sends.append(Send("contributor_agent_executor", {
            "consolidation_id": cid, "agent_name": k,
            "messages": get_step_messages(state, k, CONF_CONTRIBUTOR_READ_DEPTH, listen_depth)
        }))
    return sends

def planning_router(state: CollaborativeState) -> Literal["EXECUTE_STEP", "FINALIZE"]:
    return "EXECUTE_STEP" if state.get("current_step_index") is not None else "FINALIZE"

def debate_router(state: CollaborativeState) -> Literal["CONTINUE_DEBATE", "PLANNING"]:
    last = state["steps"][-1].get("decision", "")
    return "CONTINUE_DEBATE" if last == "CONTINUE_DEBATE" else "PLANNING"

# ==============================================================================
# 8. GRAPH
# ==============================================================================

def create_graph() -> Tuple[StateGraph, CompiledStateGraph]:
    print("--- INFO: Compiling Graph ---")
    memory = MemorySaver()
    workflow = StateGraph(CollaborativeState)

    workflow.add_node("human_input_received_node", human_input_received_node)
    workflow.add_node("lead_agent_executor", lead_agent_executor)
    workflow.add_node("debate_director_node", debate_director_node)
    workflow.add_node("consolidate_contributions_node", consolidate_contributions_node)
    workflow.add_node("contributor_agent_executor", contributor_agent_executor)
    workflow.add_node("planning_node", planning_node)
    workflow.add_node("final_response_node", final_response_node)

    workflow.set_entry_point("human_input_received_node")
    workflow.add_conditional_edges("human_input_received_node", lambda x: "END" if x["messages"][-1].content == "[END]" else "LEAD", {"END": END, "LEAD": "lead_agent_executor"})
    workflow.add_conditional_edges("lead_agent_executor", create_contributor_executors_edge, ["contributor_agent_executor"])
    workflow.add_edge("contributor_agent_executor", "consolidate_contributions_node")
    workflow.add_edge("consolidate_contributions_node", "debate_director_node")
    workflow.add_conditional_edges("debate_director_node", debate_router, {"CONTINUE_DEBATE": "lead_agent_executor", "PLANNING": "planning_node"})
    workflow.add_conditional_edges("planning_node", planning_router, {"EXECUTE_STEP": "lead_agent_executor", "FINALIZE": "final_response_node"})
    workflow.add_edge("final_response_node", END)

    return workflow, workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    workflow, graph = create_graph()
    print("Graph compiled successfully.")