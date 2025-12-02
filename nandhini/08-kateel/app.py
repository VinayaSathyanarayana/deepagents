import os
import uuid
import glob
import chainlit as cl
from dotenv import load_dotenv
from chainlit.input_widget import Select

# Import graph setup from your workflow file
from workflow import create_graph

# PDF Processing Library
from PyPDF2 import PdfReader

load_dotenv()

# Ensure API keys are set from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_available_panels():
    """
    Scans the 'agents/panels' directory for .yaml files to populate the dropdown.
    """
    # Adjust path if your folder structure is different
    path = os.path.join("agents", "panels", "*.yaml")
    files = glob.glob(path)
    
    # Extract filenames without extension (e.g., 'strategy_panel', 'investment_panel')
    panels = [os.path.basename(f).replace(".yaml", "") for f in files]
    
    # Fallback if no files found
    return panels if panels else ["strategy_panel"]

def get_coffee_break_message():
    """Generates the waiting message."""
    return "☕ The Expert Committee is convening to deliberate on your request..."

# ==============================================================================
# CHAINLIT EVENTS
# ==============================================================================

@cl.on_chat_start
async def start():
    """
    Initializes the session, creates the graph, and sets up the settings menu.
    """
    # 1. Compile graph
    _, graph = create_graph()
    
    # 2. Setup Panel Selection Settings (Dropdown)
    panels = get_available_panels()
    
    # Default to the first found panel
    default_panel = panels[0] if panels else "strategy_panel"
    
    # 3. CREATE THE DROPDOWN (Select Widget)
    # This adds the dropdown to the "Chat Settings" menu (Gear Icon)
    settings = await cl.ChatSettings(
        [
            Select(
                id="panel_name",
                label="Select Expert Panel",
                values=panels,
                initial_index=0,
            )
        ]
    ).send()

    # 4. Initialize Configuration
    # We store the selected panel in the config so the graph can access it
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "panel_name": default_panel
        }
    }

    # 5. Welcome Message with Instructions
    await cl.Message(
        content=f"**Current Panel:** `{default_panel}`\n\n"
                "👇 **To change the panel:**\n"
                "Click the **Settings Icon (⚙️)** next to the chat input and use the **Dropdown** menu."
    ).send()

    # 6. Save to session
    cl.user_session.set("graph", graph)
    cl.user_session.set("config", config)


@cl.on_settings_update
async def setup_agent(settings):
    """
    Handles the user changing the panel via the settings menu.
    """
    selected_panel = settings["panel_name"]
    
    # Update the config in the session
    config = cl.user_session.get("config")
    
    # 1. Update the Panel Name
    config["configurable"]["panel_name"] = selected_panel
    
    # 2. Generate a NEW Thread ID to reset memory for the new panel
    # This is crucial so agents don't get confused by previous conversation history from a different panel
    config["configurable"]["thread_id"] = str(uuid.uuid4())
    
    cl.user_session.set("config", config)
    
    await cl.Message(content=f"✅ **Panel switched to:** `{selected_panel}`.\n\n🔄 **Memory Reset:** The conversation history has been cleared for the new team.").send()
    
@cl.on_message
async def on_message(message: cl.Message):
    """
    Main loop: Handles user input, PDF processing, and graph execution.
    """
    config = cl.user_session.get("config")
    graph = cl.user_session.get("graph")
    
    # ------------------------------------------------------------------
    # 1. Handle PDF Uploads
    # ------------------------------------------------------------------
    context_text = ""
    if message.elements:
        for element in message.elements:
            # Check for PDF MIME type
            if "pdf" in element.mime:
                try:
                    # element.path is the temporary file path created by Chainlit
                    reader = PdfReader(element.path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    
                    # Append extracted text to context
                    # Truncating to 10k chars to prevent blowing up context window too fast
                    preview = text[:10000] 
                    context_text += f"\n\n[USER UPLOADED PDF CONTENT - {element.name}]:\n{preview}...\n"
                    
                    await cl.Message(content=f"📄 **Processed PDF:** {element.name}").send()
                except Exception as e:
                    await cl.Message(content=f"⚠️ **Error processing PDF:** {str(e)}").send()

    # Combine user's typed message with any PDF content
    full_input_content = message.content + context_text

    # ------------------------------------------------------------------
    # 2. Visual Feedback (Coffee Break)
    # ------------------------------------------------------------------
    coffee_message = cl.Message(
        author="System", 
        content=get_coffee_break_message(),
    )
    await coffee_message.send()
    
    # ------------------------------------------------------------------
    # 3. Execute Graph
    # ------------------------------------------------------------------
    ui_messages = {}
    
    # We pass "human_inputs" to trigger the entry node. 
    async for event in graph.astream_events(
        {"human_inputs": [full_input_content]}, 
        config=config, 
        version="v2"
    ):
        # Handle Custom Events (for UI logic to create placeholders)
        if event["event"] == "on_custom_event":
            agent_name = event["data"]["agent_name"]
            
            # Skip internal director logic updates from creating UI messages
            if agent_name == "debate_director":
                continue 

            # Create an empty message placeholder for the agent
            # Using the agent name as the author
            ui_message = cl.Message(author=agent_name, content=f"@{agent_name}: ", type="assistant_message")
            
            # Store message by checkpoint namespace to handle streaming correctly
            ui_messages[event["metadata"]["langgraph_checkpoint_ns"]] = ui_message
        
        # Handle Streaming Tokens
        if event["event"] == "on_chat_model_stream":
            # Find the correct message placeholder
            ui_message = ui_messages.get(event["metadata"]["langgraph_checkpoint_ns"])
            
            if ui_message:
                chunk = event["data"]["chunk"]
                
                # --- FIX FOR TOOL CALLING CRASH ---
                # Check if content is actually a string before streaming.
                # When tools are called, content might be a list or None.
                if isinstance(chunk.content, str) and chunk.content:
                    await ui_message.stream_token(token=chunk.content)
        
        # Handle Stream End (Finalize message)
        if event["event"] == "on_chat_model_end":
            ui_message = ui_messages.get(event["metadata"]["langgraph_checkpoint_ns"])
            if ui_message:
                await ui_message.send()

if __name__ == "__main__":
    cl.run()