# AI Resume Chatbot - Shreeya Gupta

A context-aware AI chatbot that understands and can intelligently answer any question about my résumé, experiences, and projects. Built as a personal portfolio guide and mock interviewer for high-stakes interviews and applications.

## 🎯 What This Does

### Core Features
- **Retrieval-Augmented Generation (RAG)**: Uses vector embeddings to index my CV, project descriptions, and personal experiences
- **Multi-Mode Interaction**: Switch between Portfolio, Interviewer, Assistant, and Technical Deep Dive modes
- **Context-Aware Responses**: Semantic search to fetch relevant information and generate personalized answers
- **Authentic Voice**: Maintains my personality and speaking style across all interactions

### Interaction Modes

#### 📋 Portfolio Mode
Perfect for recruiters and hiring managers. Walks through achievements, technical skills, and projects with detailed explanations and context.

#### 🎯 Interviewer Mode
Acts as a Google-level interviewer, asking behavioral and technical questions. Practice responses and get instant feedback.

#### 🛠️ Assistant Mode
Helps with interview practice, generates tailored cover letters, and provides personalized career advice based on my experience.

#### 🔬 Technical Deep Dive
Dive deep into technical projects, algorithms, and implementation details. Perfect for technical interviews and code reviews.

## 🛠️ Technical Implementation

### RAG Architecture
- **Vector Embeddings**: TF-IDF vectorization for semantic search
- **Cosine Similarity**: Find most relevant context for any query
- **Context Chaining**: Dynamic prompt management for different question types
- **Mode-Specific Responses**: Tailored responses based on interaction mode

### Knowledge Base
The chatbot is trained on my:
- **Personal Background**: Scorpio personality, work ethic, passions
- **Technical Skills**: Programming languages, ML algorithms, certifications
- **Projects**: VolleyVision AI, Chess Puzzle AI with detailed implementations
- **Sports Achievements**: Volleyball captain experience, leadership roles
- **Creative Work**: Published poetry, writing background
- **Social Impact**: Community service, accessibility work

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd ai-resume-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Usage
1. **Select Mode**: Choose from Portfolio, Interviewer, Assistant, or Technical Deep Dive
2. **Ask Questions**: Use suggested questions or type your own
3. **Get Responses**: Receive context-aware, personalized answers
4. **Explore Further**: Ask follow-up questions for deeper insights

## 📊 Sample Interactions

### Portfolio Mode
- "What are your key technical projects?"
- "Tell me about your leadership experience"
- "What drives your passion for AI/ML?"

### Interviewer Mode
- "Tell me about a challenging project"
- "How do you handle failure?"
- "What's your favorite AI tool?"

### Assistant Mode
- "Generate a cover letter"
- "Give me career advice"
- "Help me prepare for technical interviews"

### Technical Deep Dive
- "Explain VolleyVision AI architecture"
- "What ML algorithms did you implement?"
- "How do you approach code optimization?"

## 🎨 Personal Touch

### Scorpio Personality
- Born November 10, 2006
- Intense, passionate, and detail-oriented
- Drawn to complex problems and deep analysis
- Values authenticity and meaningful connections

### Authentic Voice
The chatbot maintains my authentic speaking style:
- Personal anecdotes and experiences
- Technical depth with practical examples
- Cross-disciplinary thinking (sports + tech + creativity)
- Philosophical insights (Dostoevsky influence)

## 🔧 Technical Features

### Smart Context Matching
- **Semantic Search**: Find relevant information using TF-IDF and cosine similarity
- **Multi-Dimensional Queries**: Handle technical, behavioral, and creative questions
- **Context Chaining**: Link related concepts across different domains

### Mode-Specific Intelligence
- **Portfolio Mode**: Professional, comprehensive responses with achievement focus
- **Interviewer Mode**: Challenging questions with follow-up prompts
- **Assistant Mode**: Helpful guidance with actionable advice
- **Technical Deep Dive**: Detailed technical explanations with code examples

### Real-Time Processing
- **Instant Responses**: No API calls, everything runs locally
- **Context Preservation**: Maintains conversation history
- **Dynamic Adaptation**: Adjusts tone and depth based on mode

## 🎯 Perfect For

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

## 🚀 Future Enhancements

### Planned Features
- **Voice Integration**: Speech-to-text and text-to-speech
- **Advanced Analytics**: Track question patterns and response effectiveness
- **Export Functionality**: Generate interview transcripts and insights
- **Multi-Language Support**: Expand to other languages

### Technical Improvements
- **Advanced RAG**: Implement more sophisticated retrieval methods
- **Real-time Learning**: Adapt responses based on interaction patterns
- **Integration APIs**: Connect with external data sources
- **Mobile App**: Native mobile application

## 🤝 Contributing

This is a personal project, but suggestions and feedback are welcome! Feel free to:
- Report bugs or issues
- Suggest new features or improvements
- Share ideas for enhancing the knowledge base

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ by Shreeya Gupta | Scorpio • November 10, 2006**

*"It takes something more than intelligence to act intelligently" - Dostoevsky* 