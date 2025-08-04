import json
import re
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class MemorySegment:
    """Represents a memory segment/folder for clustering related Q&A"""
    def __init__(self, title: str, topic: str, keywords: List[str]):
        self.title = title
        self.topic = topic
        self.keywords = keywords
        self.qa_pairs = []  # List of (question, answer) tuples
        self.embedding = None  # Sentence embedding for similarity matching
        self.created_at = None  # Timestamp for sorting
    
    def add_qa_pair(self, question: str, answer: str):
        """Add a question-answer pair to this segment"""
        self.qa_pairs.append((question, answer))
    
    def get_content(self) -> str:
        """Get all content in this segment for embedding generation"""
        content = f"{self.title} {self.topic} {' '.join(self.keywords)}"
        for q, a in self.qa_pairs:
            content += f" {q} {a}"
        return content
    
    def get_summary(self) -> str:
        """Get a summary of the segment for similarity matching"""
        return f"{self.title} {self.topic} {' '.join(self.keywords[:3])}"
    
    def to_dict(self) -> Dict:
        """Convert segment to dictionary for storage"""
        return {
            "title": self.title,
            "topic": self.topic,
            "keywords": self.keywords,
            "qa_pairs": self.qa_pairs,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create segment from dictionary"""
        segment = cls(data["title"], data["topic"], data["keywords"])
        segment.qa_pairs = data["qa_pairs"]
        if data["embedding"] is not None:
            segment.embedding = np.array(data["embedding"])
        segment.created_at = data.get("created_at")
        return segment

class MemoryClusteringSystem:
    """Advanced memory clustering system using sentence transformers"""
    def __init__(self):
        self.segments = []  # List of MemorySegment objects
        self.embedding_model = None  # Sentence transformer model
        self.similarity_threshold = 0.6  # Threshold for grouping questions
        self.is_initialized = False
        
        # Topic keywords for semantic classification
        self.topic_keywords = {
            "AI & Machine Learning": ["ai", "artificial intelligence", "machine learning", "ml", "neural networks", "deep learning", "llm", "nlp", "prompt engineering", "model", "algorithm", "cursor", "tensorflow", "sklearn"],
            "Technical Skills": ["programming", "coding", "python", "java", "javascript", "c++", "c", "html", "css", "mysql", "django", "pandas", "numpy"],
            "Projects": ["project", "volleyvision", "chess puzzle", "build", "create", "develop", "system", "application", "tool", "app"],
            "Sports & Leadership": ["volleyball", "sport", "captain", "team", "athlete", "leadership", "tournament", "medal", "championship", "coach"],
            "Books & Writing": ["book", "poetry", "poem", "write", "author", "publish", "dostoevsky", "literary", "creative", "anthology", "blue whispers", "musings"],
            "Personality & Background": ["personality", "scorpio", "passion", "philosophy", "background", "childhood", "who", "person", "character"],
            "Academic & Education": ["academic", "education", "university", "college", "certification", "course", "study", "degree", "bits", "dps"],
            "Community Service": ["volunteer", "community", "service", "counselor", "mental health", "accessibility", "cbse", "help"],
            "Temporal Memory": ["2023", "2024", "2025", "2020", "2015", "year", "when", "timeline", "evolution", "journey", "think", "thought"]
        }
    
    def initialize_model(self):
        """Initialize the sentence transformer model"""
        if not self.is_initialized:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.is_initialized = True
                print("✅ Sentence transformer model initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize sentence transformer: {e}")
                self.is_initialized = False
    
    def classify_question(self, question: str) -> str:
        """Classify question into a topic category using keyword matching"""
        question_lower = question.lower()
        
        # Calculate scores for each topic
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            topic_scores[topic] = score
        
        # Find the topic with highest score
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            if topic_scores[best_topic] > 0:
                return best_topic
        
        # Default classification based on question type
        if any(word in question_lower for word in ["what", "how", "why", "when", "where"]):
            return "General Questions"
        else:
            return "Conversation"
    
    def generate_segment_title(self, question: str, topic: str) -> str:
        """Generate a descriptive title for a memory segment"""
        question_lower = question.lower()
        
        # Extract key words from question
        words = question_lower.split()
        key_words = [word for word in words if len(word) > 3 and word not in ['what', 'when', 'where', 'which', 'about', 'shreeya', 'think', 'know', 'tell', 'me']]
        
        if key_words:
            # Use first few key words to create title
            title_words = key_words[:3]
            title = " ".join(title_words).title()
            return f"{topic}: {title}"
        else:
            return f"{topic}: {question[:30]}..."
    
    def find_similar_segment(self, question: str, topic: str) -> MemorySegment:
        """Find existing segment with similar content using sentence embeddings"""
        if not self.segments or not self.is_initialized:
            return None
        
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_model.encode([question])[0]
            
            # Find most similar segment
            best_similarity = 0
            best_segment = None
            
            for segment in self.segments:
                if segment.embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(question_embedding, segment.embedding) / (
                        np.linalg.norm(question_embedding) * np.linalg.norm(segment.embedding)
                    )
                    
                    # Check if similarity exceeds threshold and topic matches
                    if similarity > best_similarity and similarity > self.similarity_threshold:
                        # Additional check: topic should be similar
                        if segment.topic == topic or self._topics_are_related(segment.topic, topic):
                            best_similarity = similarity
                            best_segment = segment
            
            return best_segment
        except Exception as e:
            print(f"Error in similarity matching: {e}")
            return None
    
    def _topics_are_related(self, topic1: str, topic2: str) -> bool:
        """Check if two topics are related"""
        related_topics = {
            "AI & Machine Learning": ["Technical Skills", "Projects"],
            "Technical Skills": ["AI & Machine Learning", "Projects"],
            "Projects": ["AI & Machine Learning", "Technical Skills"],
            "Books & Writing": ["Personality & Background"],
            "Personality & Background": ["Books & Writing"],
            "Sports & Leadership": ["Personality & Background"],
            "Academic & Education": ["Technical Skills"],
            "Temporal Memory": ["AI & Machine Learning", "Personality & Background"]
        }
        
        return topic2 in related_topics.get(topic1, [])
    
    def create_new_segment(self, question: str, answer: str) -> MemorySegment:
        """Create a new memory segment"""
        topic = self.classify_question(question)
        title = self.generate_segment_title(question, topic)
        
        # Extract keywords from the topic
        keywords = self.topic_keywords.get(topic, [])
        
        segment = MemorySegment(title, topic, keywords)
        segment.add_qa_pair(question, answer)
        
        # Generate embedding for the segment
        if self.is_initialized:
            try:
                segment.embedding = self.embedding_model.encode([segment.get_summary()])[0]
            except Exception as e:
                print(f"Error generating embedding: {e}")
                segment.embedding = None
        
        self.segments.append(segment)
        return segment
    
    def add_qa_pair(self, question: str, answer: str) -> MemorySegment:
        """Add Q&A pair to appropriate memory segment"""
        # Initialize model if not already done
        if not self.is_initialized:
            self.initialize_model()
        
        # Try to find similar existing segment
        topic = self.classify_question(question)
        similar_segment = self.find_similar_segment(question, topic)
        
        if similar_segment:
            # Add to existing segment
            similar_segment.add_qa_pair(question, answer)
            # Update segment embedding
            if self.is_initialized:
                try:
                    similar_segment.embedding = self.embedding_model.encode([similar_segment.get_summary()])[0]
                except Exception as e:
                    print(f"Error updating embedding: {e}")
            return similar_segment
        else:
            # Create new segment
            return self.create_new_segment(question, answer)
    
    def get_segments_summary(self) -> List[Dict]:
        """Get summary of all memory segments"""
        return [
            {
                "title": segment.title,
                "topic": segment.topic,
                "qa_count": len(segment.qa_pairs),
                "keywords": segment.keywords[:5],  # First 5 keywords
                "created_at": segment.created_at
            }
            for segment in sorted(self.segments, key=lambda x: x.created_at if x.created_at else 0)
        ]
    
    def get_segment_content(self, segment_title: str) -> List[tuple]:
        """Get all Q&A pairs from a specific segment"""
        for segment in self.segments:
            if segment.title == segment_title:
                return segment.qa_pairs
        return []
    
    def save_memory(self, filename: str = "memory_segments.json"):
        """Save memory segments to file"""
        data = {
            "segments": [segment.to_dict() for segment in self.segments],
            "is_initialized": self.is_initialized
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_memory(self, filename: str = "memory_segments.json"):
        """Load memory segments from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.segments = [MemorySegment.from_dict(seg_data) for seg_data in data["segments"]]
            self.is_initialized = data.get("is_initialized", False)
            
            # Initialize model if needed
            if not self.is_initialized:
                self.initialize_model()
                
        except FileNotFoundError:
            pass  # No existing memory file
        except Exception as e:
            print(f"Error loading memory: {e}")
            self.segments = []
            self.is_initialized = False

class ResumeChatbot:
    def __init__(self):
        self.load_comprehensive_knowledge_base()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.vectorizer.fit([item['content'] for item in self.knowledge_base])
        
        # Initialize memory clustering system
        self.memory_system = MemoryClusteringSystem()
        self.memory_system.load_memory()  # Load existing memory if available
        
    def load_comprehensive_knowledge_base(self):
        """Load comprehensive knowledge base with temporal memory system"""
        
        # Enhanced temporal memory timeline with more specific year data
        self.temporal_memory = [
            {
                "time": "2015-2020",
                "content": "As a child, Shreeya was fascinated by computers, viewing them as miraculous machines. Her curiosity began with watching a short introductory video, and over the years, her love for computers deepened, inspired by her passion for mathematics. She was especially drawn to numbers, logic, and creative problem-solving. Even as a kid, she believed that computer science offered a limitless frontier for discovery and innovation."
            },
            {
                "time": "2020-2024",
                "content": "Shreeya pursued formal education in computer science. She completed foundational programming courses at Oxford Software Institute, learning C, C++, Java, and Python. She also began independently exploring areas like recursion, data structures, and basic problem-solving through multiple approaches. Her methodical mindset led her to view code as a logical and creative language, much like mathematics."
            },
            {
                "time": "2023",
                "content": "In 2023, Shreeya was deeply immersed in her academic studies and sports leadership. She was captain of Delhi's national volleyball team and was developing her technical foundation. During this time, she was focused on building her programming skills and understanding the fundamentals of computer science. She was also actively involved in community service, working as a volunteer counselor and scribe for CBSE board exams. Her interest in AI was just beginning to emerge, though she was primarily focused on mastering core programming concepts and data structures."
            },
            {
                "time": "2024",
                "content": "Shreeya began exploring machine learning, data science, and deep learning. She learned about LLMs (large language models), NLP, neural networks, and how data pipelines operate in practical AI systems. She became fluent in using tools like Google Colab, pandas, sklearn, and HuggingFace. Her focus shifted toward applied AI and understanding the inner workings of attention mechanisms, embeddings, and model interpretability."
            },
            {
                "time": "2025",
                "content": "Shreeya started building AI-powered tools herself. She explored prompt engineering, vector memory systems, chatbot design, and multimodal input. She combined her knowledge of context engineering and model behavior to create reflective agents capable of timeline-based recall. Her flagship project involved building a memory-anchored AI chatbot that could explain her own career journey across years. She also deepened her work in Java OOP, web integration, and Streamlit dashboards."
            }
        ]
        
        self.knowledge_base = [
            # Academic Background
            {
                "category": "academic",
                "content": "Currently pursuing Bachelor of Technology (B.Tech) in Computer Science Engineering from BITS Pilani, Dubai Campus. Former student of Delhi Public School, Mathura Road, New Delhi. Recognized for academic excellence and awarded for distinction in all subjects for six consecutive years. Awarded Gold Medal for Academic Excellence. Completed certifications in C & C++ (Oxford Software Institute, Grade A+), Python (Oxford Software Institute, Grade A+), Core Java (Oxford Software Institute, Grade A+). Proficient in C, C++, Python (Core & Advanced), Java (Core), JavaScript, HTML, CSS, MySQL, and Django.",
                "tags": ["academic", "education", "certifications", "skills", "BITS Pilani", "DPS"]
            },
            
            # Technical Projects & AI Context
            {
                "category": "projects",
                "content": "Built projects that blend human understanding and machine learning. 'My Understanding of AI' – a site that simplifies AI concepts using personal analogies, real research papers, and user interaction. Integrates AI-enhanced sections where users write their own AI definitions and get improvements using prompt-based NLP models. Integrates data pipelines, Apache Kafka, and Google Colab notebooks to showcase backend reasoning and ML logic. Interested in prompt engineering, contextual retrieval, multi-turn dialogues, semantic parsing, and interactive ML visualization.",
                "tags": ["AI", "ML", "projects", "prompt engineering", "NLP", "data pipelines"]
            },
            
            # VolleyVision AI Project
            {
                "category": "projects",
                "content": "VolleyVision AI is a real-time AI-powered volleyball performance analytics tool built out of deep personal need. I've played volleyball since 6th grade, captained Delhi at nationals, and participated in IPSC, BITS Sports Fest, and Heriot Watt Dubai Championships. Core features include auto-filled input system, real-time graphs (radar, pie, linear), mood prediction via linear regression, and anomaly detection with Isolation Forest. Tech stack: Python, Pandas, Scikit-learn, JS, HTML/CSS. Started with statistical feature extraction and used Isolation Forest for anomaly detection. Then layered in radar visualizations, normalized for edge cases, and later experimented with pose estimation pipelines.",
                "tags": ["VolleyVision", "volleyball", "analytics", "AI", "project", "Isolation Forest"]
            },
            
            # Sports Achievements
            {
                "category": "sports",
                "content": "Recognized multi-sport athlete with excellence in volleyball, chess, and basketball. Volleyball: Vice-captain and player in several national & university tournaments, securing Gold, Silver, and Bronze medals in major championships. Captain of the National Volleyball team of Delhi, India, Vice-Captain at Heriot Watt Dubai Tournament (Silver), participated in BITS Pilani Sports Fest (BSF largest sports festival in UAE – Silver), IPSC Nationals – Bronze (Multiple Years), and Captain at National School Games. Chess: Played for inter-university and school-level teams; secured a Bronze medal in official competitions. Also participated and medaled in Basketball and was awarded Sports Excellence Gold Medal for volleyball achievements.",
                "tags": ["volleyball", "chess", "basketball", "sports", "captain", "leadership", "medals"]
            },
            
            # Literary and Creative Work
            {
                "category": "creative",
                "content": "Published co-author of the anthologies: Blue Whispers at Midnight, Musings of Life, Poets of India, Shadows and Secrets. Performer at the Unspoken Words Open Mic, presenting original creative prose. Writing style blends mythic intimacy, emotional depth, and poetic rhythm with references to Arabic, Punjabi, and Hindi cultures. Shreeya has authored Blue Whispers of Midnight and Musings of Life — poetry collections that delve into grief, mythology, identity, and inner rebirth. Both are available on Amazon.",
                "tags": ["poetry", "writing", "anthologies", "creative", "published", "books", "Amazon"]
            },
            
            # Social Impact and Volunteering
            {
                "category": "social",
                "content": "Worked as a volunteer counselor in a national workshop for the prevention of substance abuse. Scribe for CBSE Board Exams, assisting students with disabilities. Actively involved in mental health awareness and student advocacy. Led mental health support circles for youth, volunteered in substance use awareness workshops, and worked as a scribe for CBSE board exams—helping disabled students sit critical papers.",
                "tags": ["community service", "mental health", "education", "accessibility", "volunteering", "CBSE"]
            },
            
            # Other Accomplishments
            {
                "category": "accomplishments",
                "content": "Holds Senior Diploma in Indian Classical Kathak Dance from Prayag Sangeet Samiti. Winner of ROBOTRON tech competition (First Place). Studied Media & Communication via British Council's Media Magic program. Holds FIT in Deutsch certification (A1 Grade).",
                "tags": ["dance", "Kathak", "competitions", "media", "German", "certifications"]
            },
            
            # Personal Philosophy and Work Style
            {
                "category": "personal",
                "content": "I'm Shreeya Gupta, born November 10, 2006 - a Scorpio. I'm a technophile with a poet's soul and an athletic approach to life. I believe that where there's a will, there's a way. I feel deeply and stay connected to every project I take on. I have trouble sleeping until I've asked Cursor to act like a Google dev and review my latest project at least five times. That's just who I am — passionate, relentless, and a little bit obsessed in the best way.",
                "tags": ["personality", "scorpio", "passion", "work ethic", "philosophy"]
            },
            
            # AI Tools and Development
            {
                "category": "technical",
                "content": "My favorite AI tool is Cursor AI. It's like having a genius pair programmer who just gets it. One click on 'accept,' and your bug disappears—no need to copy-paste or switch contexts. What I love most is how minimal the idea is (built on top of VS Code), yet how powerful the execution turned out to be. It gives me hope that even I can build something that impactful someday. I use Cursor AI beyond autocompletion by prompting it to generate structured testing modules, especially edge-case validation and benchmark suites. It's like pair programming, but also pair-debugging and pair-documentation.",
                "tags": ["AI tools", "Cursor", "development", "inspiration", "testing", "pair programming"]
            },
            
            # Literary Influences
            {
                "category": "creative",
                "content": "My literary influence is Fyodor Dostoevsky. 'It takes something more than intelligence to act intelligently.' His writing feels like it was written for my soul. As a Scorpio, I'm drawn to the intense, the shadowy, the spiritually charged—and Dostoevsky meets me there every time. He's less of an author and more like a best friend who bewitches me—body, mind, and soul—with every line.",
                "tags": ["literature", "Dostoevsky", "scorpio", "philosophy", "influences"]
            },
            
            # Technical Skills and Approach
            {
                "category": "technical",
                "content": "I work with multiple programming languages: C, C++, Core Java, Python (Core & Advanced), JavaScript, HTML, CSS. For Python, I use libraries like NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow. For backend development, I use Django and MySQL. I have certifications in Python & Machine Learning from Aptron Delhi, and C/C++ & Core Java from Oxford Software Institute with Grade A+. For ML model selection, I usually start by looking at the type of data I have—whether it's categorical or continuous—and what kind of problem I'm trying to solve, like regression or classification. Real-time requirements also play a big role.",
                "tags": ["programming", "languages", "certifications", "skills", "ML", "algorithms", "model selection"]
            },
            
            # Technical Philosophy and Intersections
            {
                "category": "philosophy",
                "content": "Both writing and development require clarity, structure, and emotional intelligence. When I'm building a dashboard, I think about the story it's telling—the rhythm of graphs, the tension between prediction and reality. Whether it's a line of poetry or Python, my goal is the same: evoke insight. The difference between overfitting in a model and overcoaching a player? Overfitting: memorizes data, fails to generalize. Overcoaching: reacts to every point, doesn't build intuition. In VolleyVision, I tried to keep the model performance interpretable—just like how coaches need to trust their gut, not just data.",
                "tags": ["philosophy", "creativity", "technical thinking", "analogies", "insight", "writing", "coding"]
            },
            
            # AI/ML Research and Inspiration
            {
                "category": "technical",
                "content": "Attention Is All You Need by Vaswani et al. was a turning point for me. It's not just about the architecture—it reshaped how I think about cognition and task efficiency. The idea that modern LLMs can process all input tokens simultaneously through self-attention is mind-blowing. It's like touch-typing for machines: everything flows in parallel, with zero lag between context and action. This made me deeply interested in prompt engineering. Knowing how to ask becomes just as important as what you ask. A well-crafted prompt can drastically reduce iteration time and boost output quality. It's like learning to speak fluently in the language of the model. To me, prompt design is part science, part intuition—and I think it's going to be a core ML skill moving forward.",
                "tags": ["AI", "ML", "attention mechanisms", "prompt engineering", "research", "LLMs"]
            },
            
            # VolleyVision Motivation and Scaling
            {
                "category": "projects",
                "content": "I was captain of Delhi's national volleyball team. I know how many matches were decided not by lack of skill—but lack of insight. I've seen coaches trying to manually decode patterns from paper. I built VolleyVision AI because data shouldn't be a privilege. It should be a competitive edge. For scaling this system for national-level use, I'd build a multi-team SaaS dashboard with role-based logins (coach, player, scout). Use Firebase for auth, PostgreSQL for multi-tenant storage, and integrate TensorFlow Lite for offline predictions at local tournaments.",
                "tags": ["leadership", "volleyball", "data analytics", "scalability", "insights", "motivation"]
            },
            
            # Edge Cases and Technical Challenges
            {
                "category": "technical",
                "content": "In VolleyVision AI, I handled edge cases in data visualizations. For example, in radar graphs, a player with zero attempts would break the scale—so I normalized inputs and added logic to display fallback annotations like 'Insufficient data' to avoid misleading visuals. If I had infinite compute, I'd deploy a real-time data pipeline using Kafka + Spark Streaming, and use pose estimation models like MediaPipe or OpenPose to track player movement. This would turn VolleyVision into a true in-game analytics engine, not just a post-match tool.",
                "tags": ["edge cases", "data visualization", "scalability", "technical challenges", "pose estimation"]
            },
            
            # Chess Puzzle AI Project
            {
                "category": "projects",
                "content": "Chess Puzzle AI is a web-based chess puzzle generator where users can select chess pieces and provide a board position using FEN notation. The app searches for forced mate-in-2 or mate-in-3 solutions and displays the solution with the move sequence on an interactive chessboard. Features include accepting any valid chess position via FEN notation, allowing selection of chess pieces to focus the search, automatically finding forced checkmates, and displaying solving move sequences visually. Tech stack: Streamlit, python-chess, Minimax algorithm with alpha-beta pruning, chess.svg for rendering.",
                "tags": ["Chess Puzzle AI", "chess", "algorithms", "game AI", "project", "Minimax"]
            }
        ]
        
        # Mode-specific responses
        self.mode_responses = {
            "Portfolio Mode": {
                "tone": "professional and comprehensive",
                "focus": "achievements, skills, and project details",
                "style": "detailed explanations with context"
            },
            "Interviewer Mode": {
                "tone": "challenging and evaluative",
                "focus": "behavioral questions and technical depth",
                "style": "follow-up questions and feedback"
            },
            "Assistant Mode": {
                "tone": "helpful and supportive",
                "focus": "career advice and practical guidance",
                "style": "actionable suggestions and resources"
            },
            "Technical Deep Dive": {
                "tone": "technical and analytical",
                "focus": "implementation details and algorithms",
                "style": "code examples and technical explanations"
            }
        }
    
    def find_relevant_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find most relevant context using TF-IDF and cosine similarity"""
        query_vector = self.vectorizer.transform([query])
        knowledge_vectors = self.vectorizer.transform([item['content'] for item in self.knowledge_base])
        
        similarities = cosine_similarity(query_vector, knowledge_vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.knowledge_base[i] for i in top_indices if similarities[i] > 0.1]
    
    def extract_time_reference(self, query: str) -> str:
        """Extract explicit or implicit time references from query"""
        query_lower = query.lower()
        
        # Explicit year references
        if "2025" in query or "twenty twenty five" in query_lower:
            return "2025"
        elif "2024" in query or "twenty twenty four" in query_lower:
            return "2024"
        elif "2023" in query or "twenty twenty three" in query_lower:
            return "2023"
        elif "2022" in query or "twenty twenty two" in query_lower:
            return "2022"
        elif "2021" in query or "twenty twenty one" in query_lower:
            return "2021"
        elif "2020" in query or "twenty twenty" in query_lower:
            return "2020"
        
        # Time period references
        if any(word in query_lower for word in ["child", "childhood", "kid", "young", "early"]):
            return "2015-2020"
        elif any(word in query_lower for word in ["beginning", "start", "first", "initial", "foundation"]):
            return "2020-2024"
        elif any(word in query_lower for word in ["recent", "now", "current", "latest", "today"]):
            return "2025"
        elif any(word in query_lower for word in ["evolved", "grown", "developed", "progressed", "journey"]):
            return "all"  # For evolution questions
        
        return "all"  # Default to all time periods
    
    def get_temporal_context(self, query: str) -> List[Dict]:
        """Get temporal context based on time reference in query"""
        time_ref = self.extract_time_reference(query)
        
        if time_ref == "all":
            return self.temporal_memory
        
        # Filter based on specific time period with exact year matching
        if time_ref == "2025":
            return [item for item in self.temporal_memory if "2025" in item["time"]]
        elif time_ref == "2024":
            return [item for item in self.temporal_memory if "2024" in item["time"]]
        elif time_ref == "2023":
            return [item for item in self.temporal_memory if "2023" in item["time"]]
        elif time_ref == "2020-2024":
            return [item for item in self.temporal_memory if "2020-2024" in item["time"]]
        elif time_ref == "2015-2020":
            return [item for item in self.temporal_memory if "2015-2020" in item["time"]]
        
        return self.temporal_memory
    
    def _store_in_memory(self, query: str, response: str):
        """Helper method to store Q&A pair in memory system"""
        self.memory_system.add_qa_pair(query, response)
        
        # Save memory periodically
        if len(self.memory_system.segments) % 5 == 0:  # Save every 5 new segments
            self.memory_system.save_memory()
    
    def get_response(self, query: str, mode: str = "Portfolio Mode") -> str:
        """Main method to get chatbot response with ChatGPT-style behavior and attention mechanisms"""
        
        query_lower = query.lower()
        
        # Store query for context awareness
        if not hasattr(self, 'conversation_context'):
            self.conversation_context = []
        self.conversation_context.append({"query": query, "mode": mode})
        
        # ChatGPT-style compliment handling
        compliment_words = ["cute", "cool", "awesome", "amazing", "great", "wonderful", "fantastic", "brilliant", "smart", "intelligent", "talented"]
        you_words = ["you", "you're", "you are", "your", "u", "u're", "ur"]
        if any(word in query_lower for word in compliment_words) and any(word in query_lower for word in you_words):
            response = "Thank you! That's kind of you to say. 😊"
            self._store_in_memory(query, response)
            return response
        
        # Specific Q&A pairs from the system prompt - EXACT ANSWERS
        if "favorite ai tool" in query_lower or "cursor" in query_lower:
            response = """Cursor AI — It's like having a genius pair programmer who just gets it. One click on "accept," and your bug disappears—no need to copy-paste or switch contexts. What I love most is how minimal the idea is (built on top of VS Code), yet how powerful the execution turned out to be. It gives me hope that even I can build something that impactful someday."""
            self._store_in_memory(query, response)
            return response
        
        elif "literary influence" in query_lower or "dostoevsky" in query_lower:
            response = """Fyodor Dostoevsky. "It takes something more than intelligence to act intelligently." His writing feels like it was written for my soul. As a Scorpio, I'm drawn to the intense, the shadowy, the spiritually charged—and Dostoevsky meets me there every time. He's less of an author and more like a best friend who bewitches me—body, mind, and soul—with every line."""
            self._store_in_memory(query, response)
            return response
        
        elif "volleyvision" in query_lower and "build" in query_lower and "coach notes" in query_lower:
            response = """I was captain of Delhi's national team. I know how many matches were decided not by lack of skill—but lack of insight. I've seen coaches trying to manually decode patterns from paper. I built this because data shouldn't be a privilege. It should be a competitive edge."""
            self._store_in_memory(query, response)
            return response
        
        elif "writer" in query_lower and "dev" in query_lower and "intersect" in query_lower:
            response = """Both require clarity, structure, and emotional intelligence. When I'm building a dashboard, I think about the story it's telling—the rhythm of graphs, the tension between prediction and reality. Whether it's a line of poetry or Python, my goal is the same: evoke insight."""
            self._store_in_memory(query, response)
            return response
        
        elif "overfitting" in query_lower and "overcoaching" in query_lower:
            response = """Overfitting: memorizes data, fails to generalize. Overcoaching: reacts to every point, doesn't build intuition. In VolleyVision, I tried to keep the model performance interpretable—just like how coaches need to trust their gut, not just data."""
            self._store_in_memory(query, response)
            return response
        
        elif "who is shreeya" in query_lower or "who are you" in query_lower or "personality" in query_lower or "what kind of person" in query_lower:
            response = """I'm Shreeya Gupta, born November 10, 2006 - a Scorpio. I'm a technophile with a poet's soul and an athletic approach to life. I believe that where there's a will, there's a way — and when I'm determined, I can make lemonade, lemon meringue, and lemon pie out of lemons. I feel deeply and stay connected to every project I take on. I have trouble sleeping until I've asked Cursor to act like a Google dev and review my latest project at least five times. That's just who I am — passionate, relentless, and a little bit obsessed in the best way.

As a Scorpio, I'm drawn to the intense, the shadowy, the spiritually charged. I'm someone who believes in combining technical skills with creative problem-solving. Every project is an opportunity to learn and grow. I lead by example and create environments where everyone can excel. My background spans technical development, athletic leadership, and creative expression, which gives me a unique approach to problem-solving."""
            self._store_in_memory(query, response)
            return response
        
        elif "ml model" in query_lower or "algorithm" in query_lower or "choose ml models" in query_lower:
            response = """I usually start by looking at the type of data I have—whether it's categorical or continuous—and what kind of problem I'm trying to solve, like regression or classification. Real-time requirements also play a big role. For example, I used Isolation Forest for anomaly detection because it's great with high-dimensional data and doesn't need labeled inputs. For mood prediction, I started with Linear Regression since it's fast, simple, and easy to interpret, then experimented with tree-based models to boost accuracy."""
            self._store_in_memory(query, response)
            return response
        
        # Interviewer Mode specific Q&A pairs
        elif "ml paper" in query_lower and "inspired" in query_lower:
            response = """Attention Is All You Need by Vaswani et al. was a turning point for me. It's not just about the architecture—it reshaped how I think about cognition and task efficiency. The idea that modern LLMs can process all input tokens simultaneously through self-attention is mind-blowing. It's like touch-typing for machines: everything flows in parallel, with zero lag between context and action.

This made me deeply interested in prompt engineering. Knowing how to ask becomes just as important as what you ask. A well-crafted prompt can drastically reduce iteration time and boost output quality. It's like learning to speak fluently in the language of the model. To me, prompt design is part science, part intuition—and I think it's going to be a core ML skill moving forward."""
            self._store_in_memory(query, response)
            return response
        
        elif "explain llms" in query_lower and "high schooler" in query_lower:
            response = """Imagine a super-smart student who's read almost every book, website, and article on Earth. You give them a sentence, and they try to guess the next word — not randomly, but based on everything they've ever read. That's basically how a Large Language Model (LLM) works.

Now, here's the trick: this student doesn't just read words one at a time. Instead, they look at all the words at once and decide which ones matter most right now — like highlighting the most important clues in an exam question.

This focus method is called attention — it's like having a smart highlighter in your brain. The LLM uses attention to figure out which words to 'pay attention' to before predicting the next one.

So in short: LLMs are like trained super-readers that use attention to think fast and guess smart — not just by remembering facts, but by spotting patterns in how words connect."""
            self._store_in_memory(query, response)
            return response
        
        elif "cursor" in query_lower and "beyond autocompletion" in query_lower:
            response = """I prompt it to generate structured testing modules, especially edge-case validation and benchmark suites. It's like pair programming, but also pair-debugging and pair-documentation."""
            self._store_in_memory(query, response)
            return response
        
        elif "next big thing" in query_lower and "ml" in query_lower and "excites" in query_lower:
            response = """What excites me most in machine learning right now is the rise of agentic intelligence — models that don't just predict tokens, but actually reason, plan, and make decisions over time. It's a shift from passive prediction to active problem-solving. What transformers did for sequence modeling, models like Claude 4 Opus are now doing for memory, abstraction, and agency. I'm especially fascinated by how Claude balances detailed responses with logical structure — it feels less like a tool and more like a thinking partner. I'm eager to explore Cursor's direction with agentic model integrations too — where coding becomes a conversation, not just a task. For me, making AI more human-like isn't about tricking people into thinking it's human — it's about building systems that can teach, adapt, and handle complexity the way people do. I'm excited to help shape that shift — not just building smarter tools, but actual collaborators."""
            self._store_in_memory(query, response)
            return response
        
        elif "edge cases" in query_lower and "data visualizations" in query_lower:
            response = """In radar graphs, a player with zero attempts would break the scale—so I normalized inputs and added logic to display fallback annotations like "Insufficient data" to avoid misleading visuals."""
            self._store_in_memory(query, response)
            return response
        
        elif "books" in query_lower and ("published" in query_lower or "author" in query_lower):
            response = """Shreeya has authored Blue Whispers of Midnight and Musings of Life — poetry collections that delve into grief, mythology, identity, and inner rebirth. Both are available on Amazon."""
            self._store_in_memory(query, response)
            return response
        
        # Technical Mode specific Q&A pairs
        elif "scale" in query_lower and "national" in query_lower and "system" in query_lower:
            response = """I'd build a multi-team SaaS dashboard with role-based logins (coach, player, scout). Use Firebase for auth, PostgreSQL for multi-tenant storage, and integrate TensorFlow Lite for offline predictions at local tournaments."""
            self._store_in_memory(query, response)
            return response
        
        elif "infinite compute" in query_lower and "next step" in query_lower and "volleyvision" in query_lower:
            response = """I'd deploy a real-time data pipeline using Kafka + Spark Streaming, and use pose estimation models like MediaPipe or OpenPose to track player movement. This would turn VolleyVision into a true in-game analytics engine, not just a post-match tool."""
            self._store_in_memory(query, response)
            return response
        
        elif "linear regression" in query_lower and "mood prediction" in query_lower:
            response = """It's like checking if mood follows a straight line. Linear regression gives a quick, simple way to see if things like sleep or screen time affect mood in a predictable way. It's not fancy, but it's honest — if it works, great. If not, I know I need something more complex."""
            self._store_in_memory(query, response)
            return response
        
        elif "scale volleyvision" in query_lower and "national use" in query_lower:
            response = """Like turning a local coach into a national mentor. I'd build plug-and-play tools for schools and academies — simple apps, smart video upload, cloud processing. Train models on regional data so it learns how different teams play. The goal: make it useful without needing a data scientist on the team."""
            self._store_in_memory(query, response)
            return response
        
        elif "ml debugging" in query_lower and "technique" in query_lower:
            response = """Zoom in on the data first. I pick random samples and ask — 'Does this input make sense? Is this label fair?' It's like checking ingredients before blaming the recipe. Also, I love simple plots — they often reveal what the model's really thinking."""
            self._store_in_memory(query, response)
            return response
        
        elif "imbalanced datasets" in query_lower and "handle" in query_lower:
            response = """It's like teaching a class where one student shows up 90% of the time. I balance it out by giving extra attention to the quieter group — through smarter sampling or weighted loss. But first, I ask: what matters more — catching every rare case or avoiding false alarms?"""
            self._store_in_memory(query, response)
            return response
        
        elif "how did you build volleyvision" in query_lower or "build volleyvision" in query_lower:
            response = """I started with statistical feature extraction and used Isolation Forest for anomaly detection. I then layered in radar visualizations, normalized for edge cases, and later experimented with pose estimation pipelines. The goal was to empower players and coaches with real-time strategy insights."""
            self._store_in_memory(query, response)
            return response
        
        elif "show me a poem" in query_lower or "poem" in query_lower:
            response = """Here's one that explores obsession and divinity:
"My Zahir" – https://docs.google.com/document/d/1FyOLMkmhD0Dblbyv66Nl2xwxi6y0CYg-2hP61y-zbZM/edit?usp=sharing

If you're interested in grief and family, try "The Dinner Table Dream" – https://docs.google.com/document/d/1DltglhFuG8aSl9bBzgEdxTOxxMEFEZMAR7QlicsYDMA/edit?usp=sharing"""
            self._store_in_memory(query, response)
            return response
        
        elif "tell me about your books" in query_lower:
            response = """Sure. I've authored Blue Whispers of Midnight, which explores grief and mythology, and Musings of Life, a meditation on heartbreak and self-recovery. My writing leans on sensory detail, multilingual phrasing, and spiritual undertones.

You can also read some of my individual poems:
"My Zahir" – https://docs.google.com/document/d/1FyOLMkmhD0Dblbyv66Nl2xwxi6y0CYg-2hP61y-zbZM/edit?usp=sharing
"The Dinner Table Dream" – https://docs.google.com/document/d/1DltglhFuG8aSl9bBzgEdxTOxxMEFEZMAR7QlicsYDMA/edit?usp=sharing"""
            self._store_in_memory(query, response)
            return response

        elif "tell me about your poems" in query_lower:
            response = """My poetry explores themes of obsession, divinity, grief, and family dynamics. I write with a focus on sensory detail, multilingual phrasing, and spiritual undertones that reflect my cultural background.

Here are two of my poems you can read:

"My Zahir" – https://docs.google.com/document/d/1FyOLMkmhD0Dblbyv66Nl2xwxi6y0CYg-2hP61y-zbZM/edit?usp=sharing
This poem explores obsession and divinity, drawing from my literary influences and personal philosophy.

"The Dinner Table Dream" – https://docs.google.com/document/d/1DltglhFuG8aSl9bBzgEdxTOxxMEFEZMAR7QlicsYDMA/edit?usp=sharing
This piece delves into grief and family dynamics, reflecting on loss and memory.

Both poems are part of my broader collection that includes my published books "Blue Whispers of Midnight" and "Musings of Life" available on Amazon."""
            self._store_in_memory(query, response)
            return response
        
        elif "cursor ai beyond" in query_lower or "autocompletion" in query_lower:
            response = """I prompt it to generate structured testing modules, especially edge-case validation and benchmark suites. It's like pair programming, but also pair-debugging and pair-documentation."""
            self._store_in_memory(query, response)
            return response
        
        elif "overfitting" in query_lower and "overcoaching" in query_lower:
            response = """Overfitting: memorizes data, fails to generalize. Overcoaching: reacts to every point, doesn't build intuition. In VolleyVision, I tried to keep the model performance interpretable—just like how coaches need to trust their gut, not just data."""
            self._store_in_memory(query, response)
            return response
        
        elif "ml paper" in query_lower or "inspired" in query_lower:
            response = "Attention Is All You Need by Vaswani et al. reshaped how I think about cognition. The idea that LLMs process all tokens simultaneously through self-attention is mind-blowing. This got me interested in prompt engineering — knowing how to ask is as important as what you ask."
            self._store_in_memory(query, response)
            return response
        
        elif "infinite compute" in query_lower or "next step" in query_lower:
            response = """I'd deploy a real-time data pipeline using Kafka + Spark Streaming, and use pose estimation models like MediaPipe or OpenPose to track player movement. This would turn VolleyVision into a true in-game analytics engine, not just a post-match tool."""
            self._store_in_memory(query, response)
            return response
        
        elif "writer" in query_lower and "dev" in query_lower or "writing" in query_lower and "coding" in query_lower:
            response = "Both require clarity, structure, and emotional intelligence. When building a dashboard, I think about the story it's telling. Whether poetry or Python, my goal is the same: evoke insight."
            self._store_in_memory(query, response)
            return response
        
        elif "volleyvision" in query_lower and ("build" in query_lower or "why" in query_lower or "made you" in query_lower):
            response = "I was captain of Delhi's national team. I saw how many matches were decided not by lack of skill, but lack of insight. I built this because data shouldn't be a privilege — it should be a competitive edge."
        
        # Add exact Q&A pairs for specific questions
        elif "what made you build volleyvision ai, when so many players just rely on coach notes?" in query_lower:
            response = """I was captain of Delhi's national team. I know how many matches were decided not by lack of skill—but lack of insight. I've seen coaches trying to manually decode patterns from paper. I built this because data shouldn't be a privilege. It should be a competitive edge."""
            self._store_in_memory(query, response)
            return response
        
        elif "you're a writer and a dev. how do those worlds intersect?" in query_lower:
            response = """Both require clarity, structure, and emotional intelligence. When I'm building a dashboard, I think about the story it's telling—the rhythm of graphs, the tension between prediction and reality. Whether it's a line of poetry or Python, my goal is the same: evoke insight."""
            self._store_in_memory(query, response)
            return response
        
        elif "what's a recent ml paper or idea that inspired you?" in query_lower:
            response = """Attention Is All You Need by Vaswani et al. was a turning point for me. It's not just about the architecture—it reshaped how I think about cognition and task efficiency. The idea that modern LLMs can process all input tokens simultaneously through self-attention is mind-blowing. It's like touch-typing for machines: everything flows in parallel, with zero lag between context and action.

This made me deeply interested in prompt engineering. Knowing how to ask becomes just as important as what you ask. A well-crafted prompt can drastically reduce iteration time and boost output quality. It's like learning to speak fluently in the language of the model. To me, prompt design is part science, part intuition—and I think it's going to be a core ML skill moving forward."""
            self._store_in_memory(query, response)
            return response
        
        elif "how would you explain llms to a high schooler?" in query_lower:
            response = """Imagine a super-smart student who's read almost every book, website, and article on Earth. You give them a sentence, and they try to guess the next word — not randomly, but based on everything they've ever read. That's basically how a Large Language Model (LLM) works.

Now, here's the trick: this student doesn't just read words one at a time. Instead, they look at all the words at once and decide which ones matter most right now — like highlighting the most important clues in an exam question.

This focus method is called attention — it's like having a smart highlighter in your brain. The LLM uses attention to figure out which words to 'pay attention' to before predicting the next one.

So in short: LLMs are like trained super-readers that use attention to think fast and guess smart — not just by remembering facts, but by spotting patterns in how words connect."""
            self._store_in_memory(query, response)
            return response
        
        elif "what's the next big thing in ml that excites you?" in query_lower:
            response = """What excites me most in machine learning right now is the rise of agentic intelligence — models that don't just predict tokens, but actually reason, plan, and make decisions over time. It's a shift from passive prediction to active problem-solving. What transformers did for sequence modeling, models like Claude 4 Opus are now doing for memory, abstraction, and agency. I'm especially fascinated by how Claude balances detailed responses with logical structure — it feels less like a tool and more like a thinking partner. I'm eager to explore Cursor's direction with agentic model integrations too — where coding becomes a conversation, not just a task. For me, making AI more human-like isn't about tricking people into thinking it's human — it's about building systems that can teach, adapt, and handle complexity the way people do. I'm excited to help shape that shift — not just building smarter tools, but actual collaborators."""
            self._store_in_memory(query, response)
            return response
        
        elif "how do you decide which ml model to use in a project like volleyvision ai?" in query_lower:
            response = """I usually start by looking at the type of data I have—whether it's categorical or continuous—and what kind of problem I'm trying to solve, like regression or classification. Real-time requirements also play a big role. For example, I used Isolation Forest for anomaly detection because it's great with high-dimensional data and doesn't need labeled inputs. For mood prediction, I started with Linear Regression since it's fast, simple, and easy to interpret, then experimented with tree-based models to boost accuracy."""
            self._store_in_memory(query, response)
            return response
        
        elif "what are some edge cases you've handled in your data visualizations?" in query_lower:
            response = """In radar graphs, a player with zero attempts would break the scale—so I normalized inputs and added logic to display fallback annotations like "Insufficient data" to avoid misleading visuals."""
            self._store_in_memory(query, response)
            return response
        
        elif "how would you scale this system for national-level use?" in query_lower:
            response = """I'd build a multi-team SaaS dashboard with role-based logins (coach, player, scout). Use Firebase for auth, PostgreSQL for multi-tenant storage, and integrate TensorFlow Lite for offline predictions at local tournaments."""
            self._store_in_memory(query, response)
            return response
        
        elif "if you had infinite compute, what's the next step for volleyvision ai?" in query_lower:
            response = """I'd deploy a real-time data pipeline using Kafka + Spark Streaming, and use pose estimation models like MediaPipe or OpenPose to track player movement. This would turn VolleyVision into a true in-game analytics engine, not just a post-match tool."""
            self._store_in_memory(query, response)
            return response
        
        elif "why start with linear regression for mood prediction?" in query_lower:
            response = """It's like checking if mood follows a straight line. Linear regression gives a quick, simple way to see if things like sleep or screen time affect mood in a predictable way. It's not fancy, but it's honest — if it works, great. If not, I know I need something more complex."""
            self._store_in_memory(query, response)
            return response
        
        elif "how would you scale volleyvision ai for national use?" in query_lower:
            response = """Like turning a local coach into a national mentor. I'd build plug-and-play tools for schools and academies — simple apps, smart video upload, cloud processing. Train models on regional data so it learns how different teams play. The goal: make it useful without needing a data scientist on the team."""
            self._store_in_memory(query, response)
            return response
        
        elif "what's your favorite ml debugging technique?" in query_lower:
            response = """Zoom in on the data first. I pick random samples and ask — 'Does this input make sense? Is this label fair?' It's like checking ingredients before blaming the recipe. Also, I love simple plots — they often reveal what the model's really thinking."""
            self._store_in_memory(query, response)
            return response
        
        elif "how do you handle imbalanced datasets?" in query_lower:
            response = """It's like teaching a class where one student shows up 90% of the time. I balance it out by giving extra attention to the quieter group — through smarter sampling or weighted loss. But first, I ask: what matters more — catching every rare case or avoiding false alarms?"""
            self._store_in_memory(query, response)
            return response
            self._store_in_memory(query, response)
            return response
        
        elif "scale" in query_lower and "system" in query_lower:
            response = """I'd build a multi-team SaaS dashboard with role-based logins (coach, player, scout). Use Firebase for auth, PostgreSQL for multi-tenant storage, and integrate TensorFlow Lite for offline predictions at local tournaments."""
            self._store_in_memory(query, response)
            return response
        
        elif "edge cases" in query_lower or "data visualizations" in query_lower:
            response = """In radar graphs, a player with zero attempts would break the scale—so I normalized inputs and added logic to display fallback annotations like "Insufficient data" to avoid misleading visuals."""
            self._store_in_memory(query, response)
            return response
        
        # Greeting response - ChatGPT-style engaging introduction
        elif any(word in query_lower for word in ["hi", "hello", "hey", "greetings"]):
            response = "Hey! I'm Shreeya's AI assistant. I can help you explore her AI projects, published books, volleyball experience, or technical skills. What would you like to know?"
            self._store_in_memory(query, response)
            return response
        
        # Check for general "about" questions first - give concise CV-based answers
        if any(word in query_lower for word in ["tell me about", "about shreeya", "about you", "who is", "what is"]) and not any(year in query for year in ["2023", "2024", "2025", "2020", "2015"]):
            response = "I'm Shreeya Gupta, a student at BITS Pilani Dubai Campus. I'm passionate about AI/ML, having built projects like VolleyVision AI for sports analytics. I was captain of Delhi's national volleyball team and have published two poetry books. I love combining technical skills with creative problem-solving."
            self._store_in_memory(query, response)
            return response
        
        # Check for education questions
        if any(word in query_lower for word in ["education", "university", "college", "school", "study", "degree", "course"]):
            response = "I'm currently a student at BITS Pilani Dubai Campus. I've completed foundational programming courses at Oxford Software Institute, learning C, C++, Java, and Python. I'm passionate about computer science and AI/ML."
            self._store_in_memory(query, response)
            return response
        
        # Advanced intent recognition with attention mechanisms
        intent = self._analyze_intent(query_lower)
        
        # Handle specific intents with exact CV-based answers
        if intent == "sports_inquiry":
            response = "Volleyball and Chess"
            self._store_in_memory(query, response)
            return response
        
        elif intent == "education_school":
            response = "DPS Mathura Road"
            self._store_in_memory(query, response)
            return response
        
        elif intent == "education_college":
            response = "BITS Pilani Dubai Campus"
            self._store_in_memory(query, response)
            return response
        
        elif intent == "technical_skills":
            response = "C, C++, Core Java, Python (Core + Advanced), JavaScript, HTML, CSS. Python Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow. Database: MySQL, Django."
            self._store_in_memory(query, response)
            return response
        
        elif intent == "certifications":
            response = "C/C++ and Python (Oxford Software Institute), Machine Learning & Python (Aptron Delhi)"
            self._store_in_memory(query, response)
            return response
        
        elif intent == "ai_tools":
            response = "Claude 4 Sonnet (favorite), Claude 4 Opus, Cursor AI. Used for debugging, code review, project ideation, and documentation optimization."
            self._store_in_memory(query, response)
            return response
        
        elif intent == "chess_project":
            response = "Checkmate in Code - Find the Hidden Mate chess puzzle AI. GitHub: chess-puzzle-ai. Interactive Streamlit app that generates custom chess puzzles using Stockfish engine."
            self._store_in_memory(query, response)
            return response
        
        elif intent == "volleyvision_project":
            response = "VolleyVision AI - real-time volleyball analytics tool. GitHub: Shreeya-1-pixel. Uses ML for performance analysis and mood prediction."
            self._store_in_memory(query, response)
            return response
        
        elif intent == "github_username":
            response = "Shreeya-1-pixel"
            self._store_in_memory(query, response)
            return response
        
        elif intent == "volleyball_achievements":
            response = "Captain of Delhi Nationals team, led state and national-level tournaments, recognized for leadership and strategy"
            self._store_in_memory(query, response)
            return response
        
        elif intent == "chess_achievements":
            response = "Chess Bronze Medalist (Inter-University), Zonal Basketball Champion (Gold)"
            self._store_in_memory(query, response)
            return response
        
        elif intent == "community_service":
            response = "Led mental health support circles for youth, volunteered in substance use awareness workshops, worked as scribe for CBSE board exams assisting disabled students"
            self._store_in_memory(query, response)
            return response
        
        elif intent == "out_of_scope":
            response = "This question is outside the scope of the pre-set questions. Please refer to the defined modes or ask a suggested question."
            self._store_in_memory(query, response)
            return response
        

        
        # Check for specific "think of" questions about years
        if any(word in query_lower for word in ["think of", "thought of", "think about", "thought about"]) and any(year in query for year in ["2023", "2024", "2025"]):
            if "2024" in query:
                response = "In 2024, Shreeya's focus shifted dramatically toward AI and machine learning. She began exploring ML, data science, and deep learning, learning about LLMs, NLP, and neural networks. She became fluent in tools like Google Colab, pandas, sklearn, and HuggingFace."
            elif "2023" in query:
                response = "In 2023, Shreeya was in a pivotal transition phase. She was captain of Delhi's national volleyball team while building her technical foundation. Her interest in AI was just beginning to emerge, though she was primarily focused on mastering core programming concepts and data structures."
            elif "2025" in query:
                response = "By 2025, Shreeya had evolved into a creator of AI systems. She started building AI-powered tools herself, exploring advanced concepts like prompt engineering, vector memory systems, and chatbot design. This represents her current phase where she's actively building sophisticated AI solutions."
            self._store_in_memory(query, response)
            return response
        
        # Check for temporal memory queries
        temporal_context = self.get_temporal_context(query)
        if temporal_context and len(temporal_context) > 0:
            # Special handling for AI-related temporal queries
            if ("ai" in query_lower or "artificial intelligence" in query_lower) and any(year in query for year in ["2023", "2024", "2025"]):
                response = self.generate_ai_temporal_response(query, temporal_context)
                self._store_in_memory(query, response)
                return response
            response = self.generate_temporal_response(query, temporal_context)
            self._store_in_memory(query, response)
            return response
        
        # Handle general intent or fallback responses
        if intent == "general":
            # Find relevant context for other questions
            relevant_context = self.find_relevant_context(query)
            
            if not relevant_context:
                # ChatGPT-style intent recognition for vague queries
                if any(word in query_lower for word in ["project", "work", "build", "create", "develop"]):
                    response = "I've built VolleyVision AI (volleyball analytics) and Chess Puzzle AI. Both use ML for real-time insights. Want to dive into the technical details?"
                    self._store_in_memory(query, response)
                    return response
                
                elif any(word in query_lower for word in ["book", "write", "poetry", "author", "publish"]):
                    response = "I've published 'Blue Whispers of Midnight' and 'Musings of Life' — poetry collections exploring grief, mythology, and identity. Both are on Amazon."
                    self._store_in_memory(query, response)
                    return response
                
                elif any(word in query_lower for word in ["volleyball", "sport", "captain", "team", "athlete"]):
                    response = "I was captain of Delhi's national volleyball team. That experience inspired VolleyVision AI — I saw how data could give teams a competitive edge."
                    self._store_in_memory(query, response)
                    return response
                
                elif any(word in query_lower for word in ["ai", "ml", "machine learning", "model", "algorithm"]):
                    response = "I work with AI/ML for sports analytics and mood prediction. I use Isolation Forest, pose estimation, and various ML models. Interested in specific algorithms?"
                    self._store_in_memory(query, response)
                    return response
                
                elif any(word in query_lower for word in ["tech", "stack", "language", "programming", "code"]):
                    response = "Python, Java, JavaScript, TensorFlow, scikit-learn. I love Cursor AI for pair programming. What tech interests you?"
                    self._store_in_memory(query, response)
                    return response
                
                else:
                    # ChatGPT-style intent recognition for vague queries
                    if any(word in query_lower for word in ["what", "tell", "about", "know"]):
                        response = "I'm Shreeya Gupta, a student at BITS Pilani Dubai Campus. I'm passionate about AI/ML, having built projects like VolleyVision AI. I was captain of Delhi's national volleyball team and have published two poetry books. What would you like to know about my work or experience?"
                    elif any(word in query_lower for word in ["how", "why", "when", "where"]):
                        response = "I'd be happy to explain! Could you be more specific about what you'd like to know about my work or experience?"
                    else:
                        response = "I'm here to help you learn about my AI projects, writing, sports, or tech journey. What would you like to explore?"
                    self._store_in_memory(query, response)
                    return response
        
        # Generate response based on mode and context
        response = self.generate_response(query, relevant_context, mode)
        
        # Add mode-specific follow-up with more engaging content
        if mode == "Portfolio Mode":
            response += "\n\n**Pro Tip**: Try asking about my AI projects, volleyball experience, or published books!"
        elif mode == "Interviewer Mode":
            response += "\n\n**Interview Ready**: I can discuss technical challenges, problem-solving approaches, and project scalability."
        elif mode == "Technical Deep Dive":
            response += "\n\n**Deep Dive Available**: I can explain algorithms, architecture decisions, and implementation details."
        
        # Add Q&A pair to memory clustering system
        self.memory_system.add_qa_pair(query, response)
        
        # Fit vectorizer if we have enough segments
        if len(self.memory_system.segments) > 1 and not self.memory_system.is_initialized:
            self.memory_system.initialize_model() # Initialize model for embedding
        
        # Save memory periodically
        if len(self.memory_system.segments) % 5 == 0:  # Save every 5 new segments
            self.memory_system.save_memory()
        
        return response
    
    def generate_response(self, query: str, relevant_context: List[Dict], mode: str) -> str:
        """Generate personalized response based on context and mode"""
        
        # Extract key information from context
        context_summary = "\n".join([item['content'] for item in relevant_context])
        
        # Mode-specific response generation
        if mode == "Portfolio Mode":
            return self._generate_portfolio_response(query, context_summary)
        elif mode == "Interviewer Mode":
            return self._generate_interviewer_response(query, context_summary)
        elif mode == "Technical Deep Dive":
            return self._generate_technical_response(query, context_summary)
        else:
            return self._generate_general_response(query, context_summary)
    
    def _generate_portfolio_response(self, query: str, context: str) -> str:
        """Generate professional portfolio-style responses"""
        if "technical" in query.lower() or "project" in query.lower():
            return f"Based on my experience, I've built several technical projects that showcase my skills. {context} My approach combines technical depth with practical application, always focusing on real-world impact."
        elif "leadership" in query.lower():
            return f"My leadership experience comes from captaining the Delhi national volleyball team. {context} I believe in leading by example and creating an environment where everyone can excel."
        elif "passion" in query.lower() or "drive" in query.lower():
            return f"My passion stems from seeing problems that need solving. {context} I'm particularly drawn to the intersection of sports and technology, where data can create competitive advantages."
        else:
            return f"Let me share my perspective on that. {context} My background spans technical development, athletic leadership, and creative expression, which gives me a unique approach to problem-solving."
    
    def _generate_interviewer_response(self, query: str, context: str) -> str:
        """Generate challenging interviewer-style responses"""
        if "challenging" in query.lower() or "difficult" in query.lower():
            return f"That's an interesting question. {context} Can you walk me through your thought process? What specific challenges did you face, and how did you overcome them?"
        elif "failure" in query.lower() or "setback" in query.lower():
            return f"Failure is a great teacher. {context} Tell me about a time when things didn't go as planned. What did you learn from that experience?"
        elif "strength" in query.lower():
            return f"Everyone has unique strengths. {context} What would your teammates or colleagues say is your biggest strength? How do you leverage that in your work?"
        else:
            return f"That's a thoughtful question. {context} Can you elaborate on your experience with that? I'd like to understand your approach and the outcomes you achieved."
    

    
    def _generate_technical_response(self, query: str, context: str) -> str:
        """Generate detailed technical responses"""
        # For simple questions, give concise answers
        if len(query.split()) <= 3:
            return "I can help you explore Shreeya's AI projects, published books, volleyball experience, or technical skills. What interests you most?"
        
        if "architecture" in query.lower() or "system" in query.lower():
            return f"From a technical architecture perspective, {context} I designed the system with scalability in mind. The frontend uses Chart.js for real-time visualizations, while the backend implements ML algorithms for prediction. The key was ensuring real-time performance while maintaining accuracy."
        elif "algorithm" in query.lower() or "ML" in query.lower():
            return f"Regarding the ML implementation, {context} I chose algorithms based on the specific problem requirements. For anomaly detection, Isolation Forest worked well because it doesn't require labeled data. For predictions, I started with simpler models and iterated based on performance."
        elif "optimization" in query.lower() or "performance" in query.lower():
            return f"Performance optimization was crucial. {context} I focused on efficient data structures, caching strategies, and ensuring the UI remained responsive during heavy computations. The key was balancing accuracy with real-time requirements."
        else:
            return f"From a technical standpoint, {context} I prioritize clean, maintainable code and thorough testing. The implementation details matter as much as the end result. What specific technical aspect would you like me to elaborate on?"
    
    def _generate_general_response(self, query: str, context: str) -> str:
        """Generate general responses"""
        # For simple questions, give concise answers
        if len(query.split()) <= 3:
            return "I can help you explore Shreeya's AI projects, published books, volleyball experience, or technical skills. What interests you most?"
        
        # For general questions, give CV-focused answers
        query_lower = query.lower()
        if any(word in query_lower for word in ["background", "experience", "journey", "story"]):
            return "I'm Shreeya Gupta, a student at BITS Pilani Dubai Campus. I've built AI projects like VolleyVision AI, was captain of Delhi's national volleyball team, and have published two poetry books. I love combining technical skills with creative problem-solving."
        elif any(word in query_lower for word in ["skills", "expertise", "strengths"]):
            return "My technical skills include Python, Java, JavaScript, TensorFlow, and scikit-learn. I specialize in AI/ML, sports analytics, and creative problem-solving. I also have strong leadership experience from captaining the volleyball team."
        else:
            return f"Based on my experience, {context} I believe in combining technical skills with creative problem-solving. What aspects of my background would you like to explore?"
    
    def generate_temporal_response(self, query: str, temporal_context: List[Dict]) -> str:
        """Generate temporal memory responses based on time period"""
        query_lower = query.lower()
        
        # For simple questions, give concise answers
        if len(query.split()) <= 3:
            return "I can help you explore Shreeya's journey from 2015-2025. What specific time period or aspect interests you?"
        
        # Single time period responses with enhanced detail
        if len(temporal_context) == 1:
            period = temporal_context[0]
            if "2015-2020" in period["time"]:
                return f"During her early phase (2015-2020), Shreeya was deeply inspired by the magic of computers. {period['content']} This foundational curiosity about technology and mathematics shaped her entire approach to problem-solving. Even as a child, she saw computers as miraculous machines that could unlock infinite possibilities."
            
            elif "2020-2024" in period["time"]:
                return f"Between 2020 and 2024, Shreeya built her technical foundation. {period['content']} This period was crucial for developing her methodical approach to programming and understanding the logical structure of code. She was laying the groundwork for her future AI work."
            
            elif "2023" in period["time"]:
                return "In 2023, Shreeya was in a pivotal transition phase. She was captain of Delhi's national volleyball team while building her technical foundation. Her interest in AI was just beginning to emerge, though she was primarily focused on mastering core programming concepts and data structures."
            
            elif "2024" in period["time"]:
                return "In 2024, Shreeya's focus shifted dramatically toward AI and machine learning. She began exploring ML, data science, and deep learning, learning about LLMs, NLP, and neural networks. She became fluent in tools like Google Colab, pandas, sklearn, and HuggingFace."
            
            elif "2025" in period["time"]:
                return "By 2025, Shreeya had evolved into a creator of AI systems. She started building AI-powered tools herself, exploring advanced concepts like prompt engineering, vector memory systems, and chatbot design. This represents her current phase where she's actively building sophisticated AI solutions."
        
        # Evolution/growth questions
        elif any(word in query_lower for word in ["evolved", "grown", "developed", "progressed", "journey", "over the years"]):
            return """Shreeya's journey has been a fascinating evolution of curiosity and skill:

**Early Years (2015-2020)**: As a child, she was fascinated by computers, viewing them as miraculous machines. Her curiosity began with watching a short introductory video, and over the years, her love for computers deepened, inspired by her passion for mathematics.

**Foundation Building (2020-2024)**: She pursued formal education in computer science, completing foundational programming courses at Oxford Software Institute, learning C, C++, Java, and Python. She began independently exploring recursion, data structures, and problem-solving through multiple approaches.

**Leadership & Growth (2023)**: She was captain of Delhi's national volleyball team, developing leadership skills while building her technical foundation. Her interest in AI was emerging, though she was focused on core programming concepts.

**AI Discovery (2024)**: She began exploring machine learning, data science, and deep learning. She learned about LLMs, NLP, neural networks, and became fluent in tools like Google Colab, pandas, sklearn, and HuggingFace.

**AI Creation (2025)**: She started building AI-powered tools herself, exploring advanced concepts like prompt engineering, vector memory systems, chatbot design, and multimodal input.

Her flagship project involved building a memory-anchored AI chatbot that could explain her own career journey across years - essentially creating the system you're interacting with right now! She combined her knowledge of context engineering and model behavior to create reflective agents capable of timeline-based recall.

She also deepened her work in Java OOP, web integration, and Streamlit dashboards, showing how she could bridge traditional software development with cutting-edge AI techniques. This represents her current phase where she's not just learning AI, but actively building and implementing sophisticated AI solutions that solve real-world problems."""
        
        # Default temporal response
        else:
            context_summary = " ".join([item["content"] for item in temporal_context])
            return f"Based on the relevant time period: {context_summary}"
    
    def generate_ai_temporal_response(self, query: str, temporal_context: List[Dict]) -> str:
        """Generate AI-specific temporal responses for year-based AI questions"""
        query_lower = query.lower()
        
        # For simple questions, give concise answers
        if len(query.split()) <= 3:
            return "I can help you explore Shreeya's AI journey from 2023-2025. What specific year or AI aspect interests you?"
        
        if len(temporal_context) == 1:
            period = temporal_context[0]
            
            if "2023" in period["time"]:
                return "In 2023, Shreeya's relationship with AI was in its early stages. She was primarily focused on building foundational programming skills and understanding core computer science concepts. Her interest in AI was just beginning to emerge, but she was more concentrated on developing a solid technical foundation first."
            
            elif "2024" in period["time"]:
                return "2024 was Shreeya's breakthrough year with AI! She began exploring machine learning, data science, and deep learning. She learned about LLMs, NLP, neural networks, and became fluent in tools like Google Colab, pandas, sklearn, and HuggingFace. Her focus shifted toward applied AI and understanding attention mechanisms and model interpretability."
            
            elif "2025" in period["time"]:
                return """By 2025, Shreeya had evolved from an AI learner to an AI creator! She started building AI-powered tools herself, exploring advanced concepts like prompt engineering, vector memory systems, chatbot design, and multimodal input.

Her flagship project involved building a memory-anchored AI chatbot that could explain her own career journey across years - essentially creating the system you're interacting with right now! She combined her knowledge of context engineering and model behavior to create reflective agents capable of timeline-based recall.

She also deepened her work in Java OOP, web integration, and Streamlit dashboards, showing how she could bridge traditional software development with cutting-edge AI techniques. This represents her current phase where she's not just learning AI, but actively building and implementing sophisticated AI solutions that solve real-world problems."""
        
        # For queries spanning multiple years
        else:
            return """Shreeya's AI journey has been a fascinating evolution:

**2023**: Foundation building - focused on core programming skills while AI interest was emerging
**2024**: AI discovery - learned ML, data science, LLMs, and became fluent in AI tools
**2025**: AI creation - building sophisticated AI systems and exploring advanced concepts

This progression shows her methodical approach: building strong fundamentals before diving into advanced AI, then applying that knowledge to create innovative solutions."""
    
    def get_memory_segments(self) -> List[Dict]:
        """Get all memory segments for display"""
        return self.memory_system.get_segments_summary()
    
    def get_segment_content(self, segment_title: str) -> List[tuple]:
        """Get Q&A pairs from a specific memory segment"""
        return self.memory_system.get_segment_content(segment_title)
    
    def save_memory(self):
        """Save current memory to file"""
        self.memory_system.save_memory()
    
    def _analyze_intent(self, query_lower: str) -> str:
        """Advanced intent recognition with attention mechanisms"""
        
        # Sports inquiries
        if any(word in query_lower for word in ["sports", "play", "athlete", "volleyball", "chess"]) and any(word in query_lower for word in ["shreeya", "she", "her", "what", "which"]):
            return "sports_inquiry"
        
        # Education - School
        if any(word in query_lower for word in ["school", "high school", "secondary", "dps"]) and any(word in query_lower for word in ["shreeya", "she", "her", "go", "went", "study", "where"]):
            return "education_school"
        
        # Education - College
        if any(word in query_lower for word in ["college", "university", "bachelor", "degree", "bits"]) and any(word in query_lower for word in ["shreeya", "she", "her", "go", "went", "study", "where"]):
            return "education_college"
        
        # Technical skills
        if any(word in query_lower for word in ["programming", "languages", "code", "tech stack", "python", "java", "javascript"]) and any(word in query_lower for word in ["shreeya", "she", "her", "know", "use", "what"]):
            return "technical_skills"
        
        # Certifications
        if any(word in query_lower for word in ["certifications", "certified", "oxford", "aptron"]) and any(word in query_lower for word in ["shreeya", "she", "her", "have", "what"]):
            return "certifications"
        
        # AI tools
        if any(word in query_lower for word in ["ai tools", "claude", "cursor", "favorite"]) and any(word in query_lower for word in ["shreeya", "she", "her", "use", "what"]):
            return "ai_tools"
        
        # Chess project
        if any(word in query_lower for word in ["chess", "puzzle", "checkmate"]) and any(word in query_lower for word in ["project", "github", "what"]):
            return "chess_project"
        
        # VolleyVision project
        if any(word in query_lower for word in ["volleyvision", "volleyball ai"]) and any(word in query_lower for word in ["project", "github", "what"]):
            return "volleyvision_project"
        
        # GitHub username
        if any(word in query_lower for word in ["github", "username"]) and any(word in query_lower for word in ["shreeya", "she", "her", "what"]):
            return "github_username"
        
        # Volleyball achievements
        if any(word in query_lower for word in ["volleyball", "achievements", "medals", "captain"]) and any(word in query_lower for word in ["shreeya", "she", "her", "what"]):
            return "volleyball_achievements"
        
        # Chess achievements
        if any(word in query_lower for word in ["chess", "achievements", "medals"]) and any(word in query_lower for word in ["shreeya", "she", "her", "what"]):
            return "chess_achievements"
        
        # Community service
        if any(word in query_lower for word in ["community", "service", "volunteer", "mental health"]) and any(word in query_lower for word in ["shreeya", "she", "her", "what"]):
            return "community_service"
        
        # Check for out-of-scope questions
        if any(word in query_lower for word in ["politics", "religion", "personal", "family", "relationship", "dating", "boyfriend", "girlfriend", "marriage", "children"]):
            return "out_of_scope"
        
        # Default to general response
        return "general" 