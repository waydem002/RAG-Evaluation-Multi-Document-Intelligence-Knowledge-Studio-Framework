import streamlit as st
from gtts import gTTS
import io
import os
import shutil
import base64
import pandas as pd
from src.generator import generate_podcast_script, text_to_speech_bytes
from src.engine import get_chat_engine
from src.model_loader import initialise_llm, get_embedding_model
from src.config import DATA_PATH, VECTOR_STORE_PATH, SIMILARITY_TOP_K, LLM_SYSTEM_PROMPT

# 1. Page Configuration
st.set_page_config(page_title="Multi-PDF RAG Pro", layout="wide", page_icon="📚")

# --- INITIALIZE STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "generated_script" not in st.session_state:
    st.session_state.generated_script = None

# 2. Sidebar - Controls & Hyperparameters
with st.sidebar:
    st.title("⚙️ RAG Settings")
    
    if st.button("➕ Start New Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_answer = None
        st.session_state.generated_script = None
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset()
        st.rerun()

    st.divider()

    st.subheader("📁 Document Manager")
    all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    
    if all_files:
        st.write(f"Detected **{len(all_files)}** PDFs.")
        selected_files = st.multiselect(
            "Active Documents:",
            options=all_files,
            default=all_files
        )
    else:
        st.error("No PDFs found in /data!")
        selected_files = []

    if st.button("🔄 Sync & Re-index Files", use_container_width=True):
        with st.spinner("Re-indexing..."):
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH)
            st.cache_resource.clear()
            st.rerun()

    st.divider()
    st.subheader("🔧 Configuration")
    top_k_val = st.slider("Similarity Top K", 1, 20, 20)
    
    sys_prompt_choice = st.selectbox(
        "System Prompt Role", 
        ["Default RAG", "Python Researcher", "Concise Summary"]
    )

    if "chat_engine" in st.session_state:
        if sys_prompt_choice == "Python Researcher":
            st.session_state.chat_engine.system_prompt = "You are a factual Python specialized research assistant. Answer only using context."
        else:
            st.session_state.chat_engine.system_prompt = LLM_SYSTEM_PROMPT

# 3. Engine Initialization
@st.cache_resource
def load_engine():
    llm = initialise_llm()
    embed_model = get_embedding_model()
    return get_chat_engine(llm=llm, embed_model=embed_model)

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = load_engine()

# 4. Main Chat UI
st.title("🤖 Multi-Document Intelligence")

# Display Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for s in message["sources"]:
                    st.info(s)

# 5. Chat Input & Logic
if not selected_files:
    st.warning("⚠️ Please select at least one document to begin.")
    st.stop()

if prompt := st.chat_input("Ask a question about your documents..."):
    # Reset script when a new question is asked
    st.session_state.generated_script = None
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("🔍 Searching knowledge base...", expanded=True)
        response = st.session_state.chat_engine.chat(prompt)
        
        # Capture answer for the explainer
        st.session_state.last_answer = response.response
        status.update(label="✅ Information retrieved!", state="complete", expanded=False)

        source_info = []
        for node in response.source_nodes:
            name = node.metadata.get('file_name', 'Unknown Source')
            if name in selected_files:
                source_info.append(f"📄 **File:** {name}\n\n{node.get_content()}")

        st.markdown(st.session_state.last_answer)

        if source_info:
            with st.expander("📚 Evidence & Source Citations"):
                tabs = st.tabs([f"Source {i+1}" for i in range(len(source_info))])
                for i, tab in enumerate(tabs):
                    with tab: st.markdown(source_info[i])

        st.session_state.messages.append({
            "role": "assistant", 
            "content": st.session_state.last_answer, 
            "sources": source_info
        })

# --- Inside Section 6: Knowledge Studio (app.py) ---
if st.session_state.last_answer:
    st.divider()
    with st.container(border=True):
        st.subheader("🎙️ Knowledge Studio 2.0")
        
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            if st.button("🎙️ Generate Full Podcast", width="stretch", key="v6_gen"):
                with st.spinner("Sarah and Maya are recording a deep dive... (this may take a moment)"):
                    st.session_state.generated_script = generate_podcast_script(st.session_state.last_answer)
                    st.rerun()

            if st.session_state.generated_script:
                # IMPORTANT: Check if the script was cut off
                if len(st.session_state.generated_script) < 200:
                    st.warning("The script seems a bit short. You can add more text below to reach 120 seconds!")
                
                st.session_state.generated_script = st.text_area(
                    "Sarah & Maya's Script:", 
                    value=st.session_state.generated_script, 
                    height=300, # Increased height for long scripts
                    key="v6_area"
                )
                
                if st.button("▶️ Convert to Audio", width="stretch", key="v6_audio"):
                    with st.spinner("Generating 2-minute audio... please wait..."):
                        try:
                            audio_data = text_to_speech_bytes(st.session_state.generated_script)
                            
                            import base64
                            b64 = base64.b64encode(audio_data).decode()
                            audio_html = f"""
                                <audio controls autoplay="true" style="width: 100%;">
                                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                                </audio>
                            """
                            st.markdown(audio_html, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error: {e}")

        with col2:
            st.write("🖼️ **Video Preview**")
            st.image("https://img.freepik.com/free-vector/ai-technology-concept-illustration_114360-7053.jpg", width="stretch")