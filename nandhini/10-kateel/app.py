import os
import uuid
import glob
import chainlit as cl
from datetime import datetime
from dotenv import load_dotenv
from chainlit.input_widget import Select, Slider
from PyPDF2 import PdfReader

# Import the logic for determining limits
# (We don't import MODEL_SPECS directly to avoid circular dependency if you move it, 
# but for now we rely on the workflow config to handle logic)

# ==============================================================================
# 1. SETUP
# ==============================================================================
print("--- INFO: Loading app.py ---")
try:
    from workflow import create_graph, get_current_model_spec
    print("--- INFO: Imported create_graph successfully ---")
except Exception as e:
    print(f"--- CRITICAL ERROR: Could not import workflow.py: {e} ---")
    create_graph = None

load_dotenv()

def get_available_panels():
    """Scans the agents/panels directory for configuration files."""
    path = os.path.join("agents", "panels", "*.yaml")
    files = glob.glob(path)
    panels = [os.path.basename(f).replace(".yaml", "") for f in files]
    return panels if panels else ["strategy_panel"]

# ==============================================================================
# 2. CHAINLIT EVENTS
# ==============================================================================

@cl.on_chat_start
async def start():
    if create_graph is None:
        await cl.Message(content="❌ **System Error:** `workflow.py` failed to load.").send()
        return

    try:
        _, graph = create_graph()
    except Exception as e:
        await cl.Message(content=f"❌ **Graph Error:** {e}").send()
        return

    panels = get_available_panels()
    
    # --- LOG FILE SETUP ---
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"session_{session_id}.log")

    # --- UI SETTINGS ---
    settings = await cl.ChatSettings(
        [
            Select(
                id="panel_name", 
                label="Select Expert Panel", 
                values=panels, 
                initial_index=0
            ),
            # NEW: SELECT LLM PROVIDER
            Select(
                id="active_provider",
                label="Select LLM Provider",
                values=["gemini", "openai", "anthropic", "ollama"],
                initial_index=0
            ),
            Slider(
                id="max_debate_rounds", 
                label="Debate Rounds (Per Step)", 
                initial=2, 
                min=1, 
                max=10, 
                step=1
            )
        ]
    ).send()

    # --- INITIAL CONFIGURATION ---
    # We use a default provider (gemini) for the first run logic
    default_provider = "gemini"
    
    # Simulate config to get recursion limit dynamically
    temp_config = {"configurable": {"active_provider": default_provider}}
    try:
        specs = get_current_model_spec(temp_config)
        rec_limit = specs["recursion_limit"]
    except:
        rec_limit = 1000 # Fallback

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "panel_name": panels[0],
            "max_debate_rounds": 2,
            "session_log_path": log_file_path,
            "active_provider": default_provider
        },
        "recursion_limit": rec_limit
    }

    await cl.Message(
        content=f"**Panel:** `{panels[0]}` | **Provider:** `{default_provider}`\n"
                f"📝 **Log:** `{log_file_path}`\n"
                f"🛡️ **Safety:** Limit={rec_limit}"
    ).send()

    cl.user_session.set("graph", graph)
    cl.user_session.set("config", config)

@cl.on_settings_update
async def setup_agent(settings):
    """Updates the session config when the user changes settings in the UI."""
    config = cl.user_session.get("config")
    
    # 1. Update basic settings
    provider = settings["active_provider"]
    config["configurable"]["panel_name"] = settings["panel_name"]
    config["configurable"]["max_debate_rounds"] = int(settings["max_debate_rounds"])
    config["configurable"]["active_provider"] = provider
    config["configurable"]["thread_id"] = str(uuid.uuid4()) # Reset memory
    
    # 2. Dynamically fetch new limits for the selected provider
    try:
        specs = get_current_model_spec(config)
        new_limit = specs["recursion_limit"]
        config["recursion_limit"] = new_limit
    except:
        new_limit = 1000

    cl.user_session.set("config", config)
    
    await cl.Message(
        content=f"✅ **Updated:** Provider=`{provider}` | Panel=`{settings['panel_name']}`\n"
                f"🔄 **Memory Reset** | New Limit: {new_limit}"
    ).send()
    
@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")
    graph = cl.user_session.get("graph")
    
    if not graph:
        await cl.Message(content="❌ Graph not initialized. Restart session.").send()
        return

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
    
    # Ensure recursion limit is passed at runtime
    runtime_config = config.copy()
    
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