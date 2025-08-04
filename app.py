import streamlit as st
import time
import json
import os
from resume_chatbot import ResumeChatbot

# Page configuration
st.set_page_config(
    page_title="Shreeya's AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stApp > header {display: none;}
    .stApp > footer {display: none;}
    .stApp > div[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# Custom CSS for the minimalist design
st.markdown("""
<style>
    /* Global styles with animated almond background */
    .main {
        position: relative;
        min-height: 100vh;
        color: #2F2F2F;
        font-family: 'Georgia', serif;
        overflow: hidden;
        background: linear-gradient(135deg, #EFDECD, #E6D7C3);
        animation: backgroundShift 15s ease-in-out infinite;
    }
    
    .main::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255, 255, 255, 0.05) 0%, transparent 50%);
        animation: particleFloat 20s ease-in-out infinite;
        pointer-events: none;
    }
    
    @keyframes backgroundShift {
        0%, 100% {
            background: linear-gradient(135deg, #EFDECD, #E6D7C3);
        }
        25% {
            background: linear-gradient(135deg, #E6D7C3, #D4C4A8);
        }
        50% {
            background: linear-gradient(135deg, #D4C4A8, #EFDECD);
        }
        75% {
            background: linear-gradient(135deg, #F0E6D2, #E6D7C3);
        }
    }
    
    @keyframes particleFloat {
        0%, 100% {
            transform: translateY(0px) rotate(0deg);
            opacity: 0.3;
        }
        25% {
            transform: translateY(-20px) rotate(90deg);
            opacity: 0.5;
        }
        50% {
            transform: translateY(-10px) rotate(180deg);
            opacity: 0.4;
        }
        75% {
            transform: translateY(-30px) rotate(270deg);
            opacity: 0.6;
        }
    }
    
    /* Video background */
    .video-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -2;
        object-fit: cover;
    }
    
    /* Dark overlay for readability */
    .video-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.8));
        z-index: -1;
    }
    
    /* Main container with enhanced effects */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(128, 0, 32, 0.2);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset,
            0 0 20px rgba(128, 0, 32, 0.1);
        animation: containerFloat 6s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .main-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(128, 0, 32, 0.1), transparent);
        animation: rotate 20s linear infinite;
        pointer-events: none;
    }
    
    @keyframes containerFloat {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-5px);
        }
    }
    
    @keyframes rotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    /* Header */
    .header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(245, 245, 245, 0.8));
        border-radius: 15px;
        border: 1px solid rgba(128, 0, 32, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .header h1 {
        font-size: 3rem;
        font-weight: 700;
        color: #800020;
        text-shadow: 0 0 20px rgba(128, 0, 32, 0.5);
        margin: 0;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    .header p {
        font-size: 1.2rem;
        color: #2F2F2F;
        margin: 1rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Mode selector */
    .mode-selector {
        text-align: center;
        margin-bottom: 3rem;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        border: 1px solid rgba(128, 0, 32, 0.2);
    }
    
    .mode-selector label {
        color: #800020;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: block;
    }
    
    /* Burgundy buttons with cream text and enhanced animations */
    .neon-button {
        background: linear-gradient(135deg, #800020, #600018);
        color: #F5F5DC !important;
        border: 2px solid #800020;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        margin: 0.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 0 15px rgba(128, 0, 32, 0.3);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        transform-style: preserve-3d;
        perspective: 1000px;
    }
    
    .neon-button:hover {
        background: linear-gradient(135deg, #A00028, #800020);
        color: #FFFFFF !important;
        box-shadow: 0 0 30px rgba(128, 0, 32, 0.8), 0 10px 20px rgba(0, 0, 0, 0.2);
        transform: translateY(-3px) scale(1.02) rotateX(5deg);
    }
    
    .neon-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.6s ease;
    }
    
    .neon-button:hover::before {
        left: 100%;
    }
    
    .neon-button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.2) 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: all 0.3s ease;
    }
    
    .neon-button:hover::after {
        width: 200px;
        height: 200px;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        border: 1px solid rgba(128, 0, 32, 0.3);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Chat messages */
    .chat-message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(128, 0, 32, 0.1), rgba(128, 0, 32, 0.05));
        border-color: rgba(128, 0, 32, 0.3);
        margin-left: 2rem;
        color: #2F2F2F;
    }
    
    .bot-message {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(245, 245, 245, 0.7));
        border-color: rgba(128, 0, 32, 0.2);
        margin-right: 2rem;
        color: #2F2F2F;
    }
    
    /* Input area */
    .input-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        border: 2px solid rgba(128, 0, 32, 0.3);
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Streamlit components styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #2F2F2F !important;
        border: 2px solid rgba(128, 0, 32, 0.3) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        font-family: 'Georgia', serif !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #800020 !important;
        box-shadow: 0 0 15px rgba(128, 0, 32, 0.3) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #800020, #600018) !important;
        color: #F5F5DC !important;
        border: 2px solid #800020 !important;
        border-radius: 10px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 0 15px rgba(128, 0, 32, 0.3) !important;
        transform-style: preserve-3d !important;
        perspective: 1000px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #A00028, #800020) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 30px rgba(128, 0, 32, 0.8), 0 10px 20px rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-3px) scale(1.02) rotateX(5deg) !important;
    }
    
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #2F2F2F !important;
        border: 2px solid rgba(128, 0, 32, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Suggested questions grid */
    .suggested-questions {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0, 255, 0, 0.3);
        border-radius: 50%;
        border-top-color: #00ff00;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 1rem;
        }
        
        .header h1 {
            font-size: 2rem;
        }
        
        .suggested-questions {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize chatbot
@st.cache_resource
def load_chatbot():
    return ResumeChatbot()

chatbot = load_chatbot()

# Add this after the existing imports and before the main app code
def load_memory_data():
    """Load memory data from JSON file"""
    try:
        with open('memory.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Initialize with default structure
        default_memory = {
            "Books & Writing": [
                {"Q": "What inspired 'Blue Whispers of Midnight'?", "A": "The book explores themes of grief, mythology, and inner rebirth. It was written during a period of deep reflection on loss and transformation."},
                {"Q": "Tell me about your poetry style", "A": "My poetry leans on sensory detail, multilingual phrasing, and spiritual undertones that reflect my cultural background."}
            ],
            "Sports & Leadership": [
                {"Q": "What sports do you play?", "A": "I was captain of Delhi's national volleyball team. I also play chess and basketball."},
                {"Q": "How did sports influence your leadership?", "A": "Being captain taught me about teamwork, strategy, and data-driven decision making - skills that later informed my AI projects."}
            ],
            "AI & Technology": [
                {"Q": "What's your favorite AI tool?", "A": "Cursor AI — It's like having a genius pair programmer who just gets it. One click on 'accept,' and your bug disappears."},
                {"Q": "How do you choose ML models?", "A": "I start by looking at the type of data I have and what kind of problem I'm trying to solve. Real-time requirements also play a big role."}
            ]
        }
        save_memory_data(default_memory)
        return default_memory
    except Exception as e:
        st.error(f"Error loading memory data: {e}")
        return {}

def save_memory_data(memory_data):
    """Save memory data to JSON file"""
    try:
        with open('memory.json', 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving memory data: {e}")
        return False

def get_chatbot_answer(query, memory_data):
    """Get chatbot answer from memory data"""
    query_lower = query.lower()
    
    # Search through all categories
    for category, qa_pairs in memory_data.items():
        for qa in qa_pairs:
            if query_lower in qa["Q"].lower() or any(word in query_lower for word in qa["Q"].lower().split()):
                return qa["A"]
    
    # If no direct match, return None to use the main chatbot logic
    return None

def memory_editor():
    """Shreeya's Live Memory Editor - Admin Interface"""
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h2 style="color: #800020; text-align: center; margin-bottom: 1rem;">Shreeya's Live Memory Editor</h2>
        <p style="text-align: center; color: #666; font-style: italic; margin-bottom: 2rem;">
            Admin Interface for Managing AI Knowledge Base
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Admin Authentication
    with st.container():
        st.markdown("""
        <div style="background: rgba(128, 0, 32, 0.1); padding: 1.5rem; border-radius: 10px; border: 2px solid rgba(128, 0, 32, 0.3);">
            <h4 style="color: #800020; margin-bottom: 1rem;">Admin Authentication</h4>
        </div>
        """, unsafe_allow_html=True)
        
        passcode = st.text_input("Enter Admin Passcode", type="password", placeholder="Enter passcode to access editor")
        
        if passcode != "shreeya123":
            if passcode:
                st.error("Incorrect passcode. Access denied.")
            return
    
    # Success message
    st.success("Admin access granted! Welcome to the Memory Editor.")
    
    # Load current memory data
    memory_data = load_memory_data()
    
    # Main Editor Interface
    st.markdown("""
    <div style="background: rgba(128, 0, 32, 0.05); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: #800020; margin-bottom: 1rem;">Memory Categories & Q&A Pairs</h4>
        <p style="color: #666; font-size: 0.9rem;">Edit existing Q&A pairs or add new ones to each category. Changes are saved automatically.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Track if any changes were made
    changes_made = False
    
    # Display each category
    for category in list(memory_data.keys()):
        with st.expander(f"{category} ({len(memory_data[category])} Q&A pairs)", expanded=False):
            st.markdown(f"<h5 style='color: #800020; margin-bottom: 1rem;'>{category}</h5>", unsafe_allow_html=True)
            
            # Display existing Q&A pairs
            qa_pairs_to_remove = []
            for i, qa in enumerate(memory_data[category]):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown("**Question:**")
                    new_question = st.text_input(f"Q{i+1}", value=qa["Q"], key=f"q_{category}_{i}")
                    st.markdown("**Answer:**")
                    new_answer = st.text_area(f"A{i+1}", value=qa["A"], height=100, key=f"a_{category}_{i}")
                
                with col2:
                    st.markdown("&nbsp;")  # Spacer
                    if st.button("Update", key=f"update_{category}_{i}"):
                        memory_data[category][i]["Q"] = new_question
                        memory_data[category][i]["A"] = new_answer
                        changes_made = True
                        st.success("Updated!")
                
                with col3:
                    st.markdown("&nbsp;")  # Spacer
                    if st.button("Delete", key=f"delete_{category}_{i}"):
                        qa_pairs_to_remove.append(i)
                        changes_made = True
                
                st.markdown("---")
            
            # Remove marked Q&A pairs
            for index in reversed(qa_pairs_to_remove):
                del memory_data[category][index]
            
            # Add new Q&A pair to this category
            st.markdown("""
            <div style="background: rgba(128, 0, 32, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h6 style="color: #800020; margin-bottom: 0.5rem;">Add New Q&A Pair</h6>
            </div>
            """, unsafe_allow_html=True)
            
            new_q = st.text_input("New Question", key=f"new_q_{category}")
            new_a = st.text_area("New Answer", height=100, key=f"new_a_{category}")
            
            if st.button("Add Q&A", key=f"add_{category}"):
                if new_q.strip() and new_a.strip():
                    memory_data[category].append({"Q": new_q.strip(), "A": new_a.strip()})
                    changes_made = True
                    st.success("Q&A pair added!")
                    # Clear the input fields
                    st.rerun()
                else:
                    st.error("Please enter both question and answer.")
    
    # Add new category
    st.markdown("""
    <div style="background: rgba(128, 0, 32, 0.1); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
        <h4 style="color: #800020; margin-bottom: 1rem;">Create New Category</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        new_category = st.text_input("New Category Name", placeholder="e.g., Academic Achievements")
    with col2:
        st.markdown("&nbsp;")  # Spacer
        if st.button("Create Category"):
            if new_category.strip():
                if new_category not in memory_data:
                    memory_data[new_category] = []
                    changes_made = True
                    st.success(f"Category '{new_category}' created!")
                    st.rerun()
                else:
                    st.error("Category already exists!")
            else:
                st.error("Please enter a category name.")
    
    # Save changes
    if changes_made:
        if save_memory_data(memory_data):
            st.success("All changes saved successfully!")
        else:
            st.error("Error saving changes!")
    
    # Memory Statistics
    st.markdown("""
    <div style="background: rgba(128, 0, 32, 0.05); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
        <h4 style="color: #800020; margin-bottom: 1rem;">Memory Statistics</h4>
    </div>
    """, unsafe_allow_html=True)
    
    total_categories = len(memory_data)
    total_qa_pairs = sum(len(qa_list) for qa_list in memory_data.values())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Categories", total_categories)
    with col2:
        st.metric("Total Q&A Pairs", total_qa_pairs)
    
    # Category breakdown
    st.markdown("**Category Breakdown:**")
    for category, qa_list in memory_data.items():
        st.markdown(f"• **{category}**: {len(qa_list)} Q&A pairs")

# Video background using HTML component
st.markdown("""
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -2; overflow: hidden;">
    <video autoplay muted loop playsinline style="width: 100%; height: 100%; object-fit: cover;">
        <source src="codepic.mp4" type="video/mp4">
    </video>
</div>
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.8)); z-index: -1;"></div>
""", unsafe_allow_html=True)

# Main container
st.markdown("""
<div class="main-container">
    <div class="header">
        <h1>Shreeya's AI Chatbot</h1>
        <p>Intelligent Portfolio Assistant • Powered by Advanced AI</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Mode selector
st.markdown("""
<div class="mode-selector">
    <label>Select Interaction Mode</label>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    portfolio_mode = st.button("Portfolio Mode", key="portfolio", help="Showcase projects and achievements")
with col2:
    interviewer_mode = st.button("Interviewer Mode", key="interviewer", help="Technical deep-dive discussions")
with col3:
    technical_mode = st.button("Technical Mode", key="technical", help="Advanced technical discussions")

# Initialize selected mode in session state
if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = "Portfolio Mode"

# Determine selected mode
if portfolio_mode:
    st.session_state.selected_mode = "Portfolio Mode"
elif interviewer_mode:
    st.session_state.selected_mode = "Interviewer Mode"
elif technical_mode:
    st.session_state.selected_mode = "Technical Deep Dive"

# Use the selected mode from session state
selected_mode = st.session_state.selected_mode

# Display selected mode
st.markdown(f"""
<div style="text-align: center; margin: 1rem 0; padding: 1rem; background: rgba(128, 0, 32, 0.1); border-radius: 10px; border: 1px solid rgba(128, 0, 32, 0.3);">
    <strong style="color: #800020;">Active Mode: {selected_mode}</strong>
</div>
""", unsafe_allow_html=True)

# Initialize session state for memory editor
if 'show_memory_editor' not in st.session_state:
    st.session_state.show_memory_editor = False

# Admin Tools Section
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("### Admin Tools")
    st.markdown("**MEMORY EDITOR -- ONLY FOR ADMIN**")
    if st.button("Memory Editor", help="Access Shreeya's Live Memory Editor"):
        st.session_state.show_memory_editor = True
    if st.button("Back to Chat", help="Return to main chat interface"):
        st.session_state.show_memory_editor = False

# Check if memory editor should be shown
if st.session_state.show_memory_editor:
    memory_editor()
    st.stop()  # Stop execution here to show only the memory editor

# Chat container with startup greeting
st.markdown("""
<div class="chat-container">
    <h3 style="color: #800020; text-align: center; margin-bottom: 2rem;">Chat Interface</h3>
    <div style="background: rgba(128, 0, 32, 0.1); border: 1px solid rgba(128, 0, 32, 0.3); border-radius: 10px; padding: 1rem; margin-bottom: 2rem;">
        <p style="color: #800020; margin: 0; text-align: center; font-style: italic;">
            Hi, I'm SG's AI assistant. Ask me anything about her projects, publications, or journey — from code to creativity.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Mode-specific suggested questions
st.markdown(f"""
<div style="margin: 2rem 0;">
    <h4 style="color: #800020; text-align: center; margin-bottom: 1rem;">Mode-Specific Suggested Questions</h4>
    <p style="text-align: center; color: #666; font-style: italic; margin-bottom: 1rem;">
        Questions tailored to your selected mode: {selected_mode}
    </p>
</div>
""", unsafe_allow_html=True)

# Portfolio Mode Questions
if selected_mode == "Portfolio Mode":
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Who is Shreeya?", key="who_is_shreeya"):
            st.session_state.messages.append({"role": "user", "content": "Who is Shreeya?"})
            response = chatbot.get_response("Who is Shreeya?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("What's your favorite AI tool?", key="ai_tool"):
            st.session_state.messages.append({"role": "user", "content": "What's your favorite AI tool?"})
            response = chatbot.get_response("What's your favorite AI tool?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("What's a literary influence on you?", key="literary_influence"):
            st.session_state.messages.append({"role": "user", "content": "What's a literary influence on you?"})
            response = chatbot.get_response("What's a literary influence on you?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("What made you build VolleyVision AI?", key="why_volleyvision"):
            st.session_state.messages.append({"role": "user", "content": "What made you build VolleyVision AI, when so many players just rely on coach notes?"})
            response = chatbot.get_response("What made you build VolleyVision AI, when so many players just rely on coach notes?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("How do writing and coding intersect?", key="writing_coding"):
            st.session_state.messages.append({"role": "user", "content": "You're a writer and a dev. How do those worlds intersect?"})
            response = chatbot.get_response("You're a writer and a dev. How do those worlds intersect?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("Overfitting vs Overcoaching?", key="overfitting_overcoaching"):
            st.session_state.messages.append({"role": "user", "content": "What's the difference between overfitting in a model and overcoaching a player?"})
            response = chatbot.get_response("What's the difference between overfitting in a model and overcoaching a player?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Interviewer Mode Questions
elif selected_mode == "Interviewer Mode":
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Recent ML paper that inspired you?", key="ml_paper"):
            st.session_state.messages.append({"role": "user", "content": "What's a recent ML paper or idea that inspired you?"})
            response = chatbot.get_response("What's a recent ML paper or idea that inspired you?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("Explain LLMs to a high schooler", key="explain_llms"):
            st.session_state.messages.append({"role": "user", "content": "How would you explain LLMs to a high schooler?"})
            response = chatbot.get_response("How would you explain LLMs to a high schooler?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("How do you use Cursor AI beyond autocompletion?", key="cursor_beyond"):
            st.session_state.messages.append({"role": "user", "content": "How do you use Cursor AI beyond autocompletion?"})
            response = chatbot.get_response("How do you use Cursor AI beyond autocompletion?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("What's the next big thing in ML?", key="next_big_ml"):
            st.session_state.messages.append({"role": "user", "content": "What's the next big thing in ML that excites you?"})
            response = chatbot.get_response("What's the next big thing in ML that excites you?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("How do you choose ML models?", key="choose_ml_models"):
            st.session_state.messages.append({"role": "user", "content": "How do you decide which ML model to use in a project like VolleyVision AI?"})
            response = chatbot.get_response("How do you decide which ML model to use in a project like VolleyVision AI?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("Edge cases in data visualizations?", key="edge_cases"):
            st.session_state.messages.append({"role": "user", "content": "What are some edge cases you've handled in your data visualizations?"})
            response = chatbot.get_response("What are some edge cases you've handled in your data visualizations?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Technical Mode Questions
elif selected_mode == "Technical Deep Dive":
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Scale VolleyVision for national use", key="scale_national"):
            st.session_state.messages.append({"role": "user", "content": "How would you scale this system for national-level use?"})
            response = chatbot.get_response("How would you scale this system for national-level use?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("Infinite compute next step for VolleyVision", key="infinite_compute"):
            st.session_state.messages.append({"role": "user", "content": "If you had infinite compute, what's the next step for VolleyVision AI?"})
            response = chatbot.get_response("If you had infinite compute, what's the next step for VolleyVision AI?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("Why Linear Regression for mood prediction?", key="linear_regression"):
            st.session_state.messages.append({"role": "user", "content": "Why start with Linear Regression for mood prediction?"})
            response = chatbot.get_response("Why start with Linear Regression for mood prediction?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("Scale VolleyVision for national use (detailed)", key="scale_detailed"):
            st.session_state.messages.append({"role": "user", "content": "How would you scale VolleyVision AI for national use?"})
            response = chatbot.get_response("How would you scale VolleyVision AI for national use?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("Favorite ML debugging technique?", key="debugging"):
            st.session_state.messages.append({"role": "user", "content": "What's your favorite ML debugging technique?"})
            response = chatbot.get_response("What's your favorite ML debugging technique?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("Handle imbalanced datasets?", key="imbalanced"):
            st.session_state.messages.append({"role": "user", "content": "How do you handle imbalanced datasets?"})
            response = chatbot.get_response("How do you handle imbalanced datasets?", selected_mode)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()



# Chat input
st.markdown("""
<div class="input-container">
    <h4 style="color: #800020; margin-bottom: 1rem;">Ask me anything...</h4>
</div>
""", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with typing animation
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot.get_response(prompt, selected_mode)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Memory Segments Display
st.markdown("""
<div style="margin: 2rem 0;">
    <h4 style="color: #800020; text-align: center; margin-bottom: 1rem;">Memory Segments</h4>
    <p style="text-align: center; color: #666; font-style: italic; margin-bottom: 2rem;">
        Related questions are automatically grouped into topic-based memory segments
    </p>
</div>
""", unsafe_allow_html=True)

# Get memory segments
memory_segments = chatbot.get_memory_segments()

if memory_segments:
    # Group segments by topic
    topic_groups = {}
    for segment in memory_segments:
        topic = segment['topic']
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(segment)
    
    # Display segments grouped by topic
    for topic, segments in topic_groups.items():
        # Calculate total Q&A pairs for this topic
        total_qa_pairs = sum(segment['qa_count'] for segment in segments)
        
        # Get all Q&A pairs for this topic
        all_qa_pairs = []
        for segment in segments:
            qa_pairs = chatbot.get_segment_content(segment['title'])
            all_qa_pairs.extend(qa_pairs)
        
        # Get the first question to show in the expander title
        first_question = all_qa_pairs[0][0] if all_qa_pairs else "No questions yet"
        # Truncate long questions for cleaner display
        display_question = first_question[:60] + "..." if len(first_question) > 60 else first_question
        
        with st.expander(f"{topic}: {display_question} ({total_qa_pairs} Q&A pairs)", expanded=False):
            # Display Q&A pairs for this topic
            for i, (question, answer) in enumerate(all_qa_pairs, 1):
                st.markdown(f"""
                <div style="margin: 1rem 0; padding: 1rem; background: rgba(255, 255, 255, 0.8); border-radius: 8px; border-left: 3px solid #800020;">
                    <p style="margin: 0 0 0.5rem 0;"><strong>Q{i}:</strong> {question}</p>
                    <p style="margin: 0; color: #666;"><strong>A{i}:</strong> {answer}</p>
                </div>
                """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(128, 0, 32, 0.1); border-radius: 15px; border: 2px dashed rgba(128, 0, 32, 0.3);">
        <p style="color: #800020; margin: 0; font-style: italic; font-size: 1.1rem;">
            No memory segments yet. Start asking questions to build your knowledge base!
        </p>
        <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Related questions will be automatically grouped into topic-based segments
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; background: rgba(255, 255, 255, 0.9); border-radius: 15px; border: 1px solid rgba(128, 0, 32, 0.2);">
    <p style="color: #2F2F2F; margin: 0;">Powered by Advanced AI • Built with Streamlit • Designed for Excellence</p>
</div>
""", unsafe_allow_html=True) 

# Initialize session state
if 'show_memory_editor' not in st.session_state:
    st.session_state.show_memory_editor = False

# Check if memory editor should be shown
if st.session_state.show_memory_editor:
    memory_editor()
    st.stop()  # Stop execution here to show only the memory editor 