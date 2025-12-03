import os
import uuid
import glob
import chainlit as cl
from dotenv import load_dotenv
from chainlit.input_widget import Select, Slider
from PyPDF2 import PdfReader

# ==============================================================================
# 🛠️ USER CONFIGURATION (EDIT UI SETTINGS HERE)
# ==============================================================================

# --- EXECUTION SAFETY ---
# CRITICAL: Controls how many graph steps (nodes) can run before halting.
# Multi-step plans generate ~10 steps per plan item. A 10-step plan needs ~100+ limit.
CONF_RECURSION_LIMIT = 100     

# --- UI DEFAULTS ---
CONF_DEFAULT_DEBATE_ROUNDS = 2   # Default setting for the Slider
CONF_MAX_DEBATE_ROUNDS = 10      # Max setting for the Slider

# ==============================================================================
# 1. SETUP
# ==============================================================================

print("--- INFO: Loading app.py ---")
try:
    from workflow import create_graph
    print("--- INFO: Imported create_graph successfully ---")
except Exception as e:
    print(f"--- CRITICAL ERROR: Could not import workflow.py: {e} ---")
    create_graph = None

load_dotenv()

def get_available_panels():
    path = os.path.join("agents", "panels", "*.yaml")
    files = glob.glob(path)
    panels = [os.path.basename(f).replace(".yaml", "") for f in files]
    return panels if panels else ["strategy_panel"]

# ==============================================================================
# 2. CHAINLIT EVENTS
# ==============================================================================

@cl.on_chat_start
async def start():
    print("--- INFO: Chat Start Triggered ---")
    if create_graph is None:
        await cl.Message(content="❌ **System Error:** `workflow.py` failed to load.").send()
        return

    try:
        _, graph = create_graph()
    except Exception as e:
        await cl.Message(content=f"❌ **Graph Error:** {e}").send()
        return

    panels = get_available_panels()
    default_panel = panels[0] if panels else "strategy_panel"
    
    settings = await cl.ChatSettings(
        [
            Select(id="panel_name", label="Select Expert Panel", values=panels, initial_index=0),
            Slider(
                id="max_debate_rounds", 
                label="Debate Rounds (Per Step)", 
                initial=CONF_DEFAULT_DEBATE_ROUNDS, 
                min=1, 
                max=CONF_MAX_DEBATE_ROUNDS, 
                step=1
            )
        ]
    ).send()

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "panel_name": default_panel,
            "max_debate_rounds": CONF_DEFAULT_DEBATE_ROUNDS
        }
    }

    await cl.Message(
        content=f"**Panel:** `{default_panel}` | **System:** Plan-Execute Ready (Limit: {CONF_RECURSION_LIMIT})"
    ).send()

    cl.user_session.set("graph", graph)
    cl.user_session.set("config", config)

@cl.on_settings_update
async def setup_agent(settings):
    config = cl.user_session.get("config")
    config["configurable"]["panel_name"] = settings["panel_name"]
    config["configurable"]["max_debate_rounds"] = int(settings["max_debate_rounds"])
    config["configurable"]["thread_id"] = str(uuid.uuid4())
    
    cl.user_session.set("config", config)
    await cl.Message(content=f"✅ **Settings Updated:** `{settings['panel_name']}`").send()
    
@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")
    graph = cl.user_session.get("graph")
    
    if not graph:
        await cl.Message(content="❌ Graph not initialized. Restart session.").send()
        return

    # PDF Handling
    context_text = ""
    if message.elements:
        for element in message.elements:
            if "pdf" in element.mime:
                try:
                    reader = PdfReader(element.path)
                    text = "".join([p.extract_text() for p in reader.pages])[:10000]
                    context_text += f"\n\n[PDF: {element.name}]:\n{text}...\n"
                    await cl.Message(content=f"📄 **Processed:** {element.name}").send()
                except:
                    await cl.Message(content=f"⚠️ **Error:** Could not read PDF.").send()

    await cl.Message(author="System", content="☕ The Expert Committee is working...").send()
    
    ui_messages = {}
    
    # --- FIX: MERGE RECURSION LIMIT HERE ---
    # We ensure the recursion limit is present in the runtime config, 
    # overriding any potential stale session data.
    runtime_config = config.copy()
    runtime_config["recursion_limit"] = CONF_RECURSION_LIMIT
    
    async for event in graph.astream_events(
        {"human_inputs": [message.content + context_text]}, 
        config=runtime_config, 
        version="v2"
    ):
        if event["event"] == "on_custom_event":
            name = event["data"]["agent_name"]
            if name in ["debate_director"]: continue
            if name == "Planner":
                await cl.Message(author="System", content="📋 **Planning Phase Active...**", type="system_message").send()
                continue
            
            ui_messages[event["metadata"]["langgraph_checkpoint_ns"]] = cl.Message(
                author=name, content=f"@{name}: ", type="assistant_message"
            )
        
        if event["event"] == "on_chat_model_stream":
            ui_msg = ui_messages.get(event["metadata"]["langgraph_checkpoint_ns"])
            if ui_msg:
                chunk = event["data"]["chunk"]
                content = chunk.content
                if isinstance(content, str) and content:
                    await ui_msg.stream_token(token=content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_val = item.get("text", "")
                            if text_val: await ui_msg.stream_token(token=text_val)
        
        if event["event"] == "on_chat_model_end":
            ui_msg = ui_messages.get(event["metadata"]["langgraph_checkpoint_ns"])
            if ui_msg: await ui_msg.send()

if __name__ == "__main__":
    cl.run()