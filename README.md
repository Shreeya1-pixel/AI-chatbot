# Shreeya's Memory-Driven AI Chatbot
https://ai-chatbot-iicyqqjjdfov7bedjjawit.streamlit.app
A sophisticated AI chatbot that doesn't just answer questions—it remembers, evolves, and thinks like me. Built for high-stakes interviews, portfolio showcases, and authentic conversations that capture the real Shreeya Gupta experience.

## Key Features That Set This Apart

### 1. Chronological Memory Recall (Year-Specific Insight)
Ask: "What did Shreeya think about AI in 2023?" or "What was her focus in 2021?"
The bot uses anchored temporal memory to simulate how my mindset, interests, and skills evolved over time. It enables timeline coherence across life events, reflections, and career milestones—just like having a conversation with someone who actually lived through those years.

### 2. Mode-Specific Persona Switching
Users select from intelligent modes:
- **Technical Mode**: ML projects, Java, VolleyVision AI
- **Portfolio Mode**: Awards, writing, outreach  
- **Interviewer Mode**: Uses exact user-fed Q&A pairs—no AI improvisation

### 3. CV-Locked Answer Precision
Factual answers are locked based on CV inputs:
- "School?" → DPS Mathura Road
- "College?" → BITS Pilani Dubai Campus
- "Sports?" → Volleyball and Chess
- "Achievements?" → Fully listed if asked

No hallucination. No paraphrasing. Pure accuracy.

### 4. Real-Time Admin Memory Editor (Backend-Connected)
Admin-only, password-protected backend allows live updates to Q&A data, CV entries, and memory blocks. No model retraining required. Ideal for interviews, applications, and evolving CVs.

### 5. Auto-Categorized Memory Segments (ChatGPT-style Grouping)
Every Q&A pair is stored under an intelligent category label: Education, Writing, AI Projects, Sports, Literary Influences, etc. Responses are dynamically grouped to maintain conversational clarity and context. Perfect for long chats, follow-up chaining, and user-driven exploration.

### 6. Entity-Aware Context + Follow-Up Memory Locking
Remembers context across questions:
- "Where did she go to college?" → "BITS Pilani Dubai Campus"
- "What did she study there?" → Responds with correct degree

Maintains identity and context threading without confusion. Corrects user misunderstandings by reverting to memory base.

### 7. LLM Stack and Toolchain
Powered by Cursor AI and advanced LLM core with context memory engine for consistent answers. Backend integration for admin tools (Flask/Django/Node-ready). Modular prompt routing and fallback logic. Designed for Google-level technical clarity and creative depth.

## Technical Architecture

### Memory-Driven Design
Built around a sophisticated memory system that captures not just facts, but the evolution of thoughts and experiences over time. Uses sentence transformers for semantic similarity and cosine similarity for context matching.

### Knowledge Base Structure
The system maintains multiple layers of knowledge:
- **Temporal Memory**: Year-specific insights and mindset evolution
- **CV-Locked Facts**: Hardcoded factual information for zero hallucination
- **Project Details**: Deep technical implementations and decision rationale
- **Personality Framework**: Authentic voice and response patterns

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/Shreeya1-pixel/AI-chatbot.git
cd AI-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Usage
1. Select Mode: Choose from Portfolio, Interviewer, or Technical Deep Dive
2. Ask Questions: Use suggested questions or type your own
3. Get Responses: Receive context-aware, personalized answers
4. Explore Further: Ask follow-up questions for deeper insights

## Sample Interactions

### Portfolio Mode
- "What are your key technical projects?"
- "Tell me about your leadership experience"
- "What drives your passion for AI/ML?"

### Interviewer Mode
- "Tell me about a challenging project"
- "How do you handle failure?"
- "What's your favorite AI tool?"

### Technical Deep Dive
- "Explain VolleyVision AI architecture"
- "What ML algorithms did you implement?"
- "How do you approach code optimization?"

## Personal Touch

### Authentic Voice
The chatbot maintains my authentic speaking style and personality. Born November 10, 2006, I'm intense, passionate, and detail-oriented—drawn to complex problems and deep analysis. The system captures this authentic voice across all interactions.
- Personal anecdotes and experiences
- Technical depth with practical examples
- Cross-disciplinary thinking (sports + tech + creativity)
- Philosophical insights (Dostoevsky influence)

## Perfect For

### Job Applications
- Practice interview questions
- Generate tailored responses
- Understand your strengths and experiences

### Portfolio Presentations
- Walk through projects with technical depth
- Explain leadership experience with examples
- Showcase cross-disciplinary thinking

### Career Development
- Get personalized career advice
- Identify areas for improvement
- Plan next steps in your journey

## Contributing

This is a personal project, but suggestions and feedback are welcome! Feel free to:
- Report bugs or issues
- Suggest new features or improvements
- Share ideas for enhancing the knowledge base

## License

This project is open source and available under the MIT License.

---

**Built by Shreeya Gupta | November 10, 2006**

*"It takes something more than intelligence to act intelligently" - Dostoevsky* 
