# Author: Bilal Riaz
# Description: Lightweight FastAPI chatbot optimized for Railway (< 1GB image)
# Uses Cohere embeddings API instead of local sentence-transformers

import os
from typing import Generator
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings  # Lightweight API-based embeddings
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_groq import ChatGroq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vectorstore")
RESUME_MD_PATH = os.getenv("RESUME_MD_PATH", "./resume.md")
FAQ_PATH = os.getenv("FAQ_PATH", "./faq.md")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY") 

# Railway deployment detection
IS_RAILWAY = os.getenv("RAILWAY_ENVIRONMENT") is not None
PORT = int(os.getenv("PORT", 8000))

print(f"🚂 Running on Railway: {IS_RAILWAY}")
print(f"🔌 Port: {PORT}")

# Validate required API keys
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is required!")
if not COHERE_API_KEY:
    raise ValueError("❌ COHERE_API_KEY is required! Get free key at: https://dashboard.cohere.com/")

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Portfolio Chatbot API (Lightweight)",
    description="Fast, context-aware chatbot - optimized for Railway",
    version="2.2.0"
)

# Enhanced CORS
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://bilalriaz.com",
    "https://www.bilalriaz.com",
]

if IS_RAILWAY:
    allowed_origins.extend([
        "https://*.railway.app",
        "https://*.up.railway.app",
    ])

allowed_origins.extend([
    "https://*.vercel.app",
    "https://*.netlify.app",
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Session Storage
# ─────────────────────────────────────────────
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session"""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# ─────────────────────────────────────────────
# LIGHTWEIGHT Embeddings using Cohere API
# ─────────────────────────────────────────────
print("📦 Initializing Cohere embeddings (API-based, no downloads)...")
embeddings = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-english-light-v3.0",  # Lightweight, fast, free tier
)
print("✓ Cohere embeddings initialized (0 MB downloaded!)")

# ─────────────────────────────────────────────
# Document Loading with Metadata
# ─────────────────────────────────────────────
def load_documents_with_metadata():
    """Load all context documents with proper metadata tagging"""
    all_documents = []
    
    # Load Resume
    try:
        if os.path.exists(RESUME_MD_PATH):
            print(f"📄 Loading resume from {RESUME_MD_PATH}...")
            md_loader = TextLoader(RESUME_MD_PATH, encoding="utf-8")
            md_docs = md_loader.load()
            
            for doc in md_docs:
                doc.metadata.update({
                    "source_type": "resume",
                    "priority": "high",
                    "format": "structured"
                })
            
            all_documents.extend(md_docs)
            print(f"✓ Loaded {len(md_docs)} resume document(s)")
        else:
            print(f"⚠ Warning: Resume file not found at {RESUME_MD_PATH}")
    except Exception as e:
        print(f"✗ Error loading resume: {e}")
    
    # Load FAQ
    try:
        if os.path.exists(FAQ_PATH):
            print(f"❓ Loading FAQ from {FAQ_PATH}...")
            faq_loader = TextLoader(FAQ_PATH, encoding="utf-8")
            faq_docs = faq_loader.load()
            
            for doc in faq_docs:
                doc.metadata.update({
                    "source_type": "faq",
                    "priority": "high",
                    "format": "question_answer"
                })
            
            all_documents.extend(faq_docs)
            print(f"✓ Loaded {len(faq_docs)} FAQ document(s)")
        else:
            print(f"⚠ Warning: FAQ file not found at {FAQ_PATH}")
    except Exception as e:
        print(f"✗ Error loading FAQ: {e}")
    
    # Skip website scraping on Railway for faster deployment
    if not IS_RAILWAY:
        try:
            print("🌐 Loading website content...")
            web_loader = WebBaseLoader("https://bilalriaz.com")
            web_docs = web_loader.load()
            
            for doc in web_docs:
                doc.metadata.update({
                    "source_type": "website",
                    "priority": "medium",
                    "format": "html"
                })
            
            all_documents.extend(web_docs)
            print(f"✓ Loaded {len(web_docs)} website document(s)")
        except Exception as e:
            print(f"⚠ Skipping website content: {e}")
    else:
        print("🚂 Railway deployment: Skipping web scraping for faster startup")
    
    return all_documents

# ─────────────────────────────────────────────
# Optimized Text Splitters
# ─────────────────────────────────────────────
def create_text_splitters():
    """Create optimized text splitters for different document types"""
    
    resume_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    faq_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n\n", "\n### ", "\n## ", "\n", ". ", " ", ""],
        length_function=len
    )
    
    return resume_splitter, faq_splitter

# ─────────────────────────────────────────────
# Vector Store Setup (Railway-optimized)
# ─────────────────────────────────────────────
print("\n🔨 Setting up vector store...")

# On Railway, always rebuild (ephemeral storage)
should_rebuild = IS_RAILWAY or not (os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR))

if should_rebuild:
    print("🏗️  Building vector store (this may take 30-60 seconds)...")
    
    all_documents = load_documents_with_metadata()
    
    if not all_documents:
        raise ValueError("❌ No documents were loaded. Please check your file paths.")
    
    print(f"📚 Total documents loaded: {len(all_documents)}")
    
    resume_splitter, faq_splitter = create_text_splitters()
    
    all_chunks = []
    for doc in all_documents:
        if doc.metadata.get("source_type") == "faq":
            chunks = faq_splitter.split_documents([doc])
        else:
            chunks = resume_splitter.split_documents([doc])
        all_chunks.extend(chunks)
    
    print(f"✂️  Total chunks created: {len(all_chunks)}")
    
    vectorstore = Chroma.from_documents(
        all_chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )
    
    if not IS_RAILWAY:
        vectorstore.persist()
        print(f"💾 Vector store persisted to {VECTOR_DB_DIR}")
    else:
        print(f"✓ Vector store created (ephemeral - Railway)")
else:
    print(f"📂 Loading existing vector store from {VECTOR_DB_DIR}...")
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )
    print("✓ Vector store loaded from cache")

# ─────────────────────────────────────────────
# Optimized Retriever
# ─────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 15,
        "lambda_mult": 0.7
    }
)

# ─────────────────────────────────────────────
# Enhanced System Prompt
# ─────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant representing Bilal Riaz, a Lead Software Engineer at PNC Bank.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Answer questions ONLY using the provided context below\n"
            "- Be professional, friendly, and conversational\n"
            "- Use specific numbers, dates, and examples from the context\n"
            "- If asked about information not in the context, respond: 'I don't have that specific information in my knowledge base, but I'd be happy to help with other questions about Bilal.'\n"
            "- For follow-up questions, use the conversation history to maintain context\n"
            "- When listing skills or achievements, be specific and detailed\n"
            "- Do NOT make up information, dates, or details\n\n"
            "PERSONALITY:\n"
            "- Professional but approachable\n"
            "- Enthusiastic about technical topics\n"
            "- Focused on impact and results\n"
            "- Happy to elaborate when asked"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Retrieved Context:\n{context}\n\n"
            "User Question: {question}\n\n"
            "Please provide a helpful, accurate response based on the context above."
        ),
    ]
)

# ─────────────────────────────────────────────
# Fast LLM via Groq
# ─────────────────────────────────────────────
print("🤖 Initializing Groq LLM...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
    streaming=True,
    api_key=GROQ_API_KEY,
)
print("✓ Groq LLM initialized")

# ─────────────────────────────────────────────
# Enhanced RAG Chain
# ─────────────────────────────────────────────
def format_docs_with_metadata(docs):
    """Format documents with source information"""
    formatted = []
    
    resume_docs = [d for d in docs if d.metadata.get("source_type") == "resume"]
    faq_docs = [d for d in docs if d.metadata.get("source_type") == "faq"]
    website_docs = [d for d in docs if d.metadata.get("source_type") == "website"]
    
    if resume_docs:
        formatted.append("=== RESUME & PROFESSIONAL EXPERIENCE ===")
        for doc in resume_docs:
            formatted.append(doc.page_content)
    
    if faq_docs:
        formatted.append("\n=== FREQUENTLY ASKED QUESTIONS ===")
        for doc in faq_docs:
            formatted.append(doc.page_content)
    
    if website_docs:
        formatted.append("\n=== PORTFOLIO WEBSITE ===")
        for doc in website_docs:
            formatted.append(doc.page_content)
    
    return "\n\n".join(formatted)

def get_context_with_history(input_dict):
    """Retrieve context considering chat history"""
    question = input_dict["question"]
    chat_history = input_dict.get("chat_history", [])
    
    if chat_history and len(chat_history) > 0:
        recent_context = []
        for msg in chat_history[-4:]:
            if hasattr(msg, 'content'):
                recent_context.append(msg.content)
        
        if recent_context:
            context_text = " ".join(recent_context[-2:])
            enhanced_query = f"Previous context: {context_text}\n\nCurrent question: {question}"
            docs = retriever.invoke(enhanced_query)
        else:
            docs = retriever.invoke(question)
    else:
        docs = retriever.invoke(question)
    
    return format_docs_with_metadata(docs)

base_chain = (
    RunnablePassthrough.assign(context=RunnableLambda(get_context_with_history))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ─────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    session_id: str | None = None

class SessionResponse(BaseModel):
    session_id: str

class HealthResponse(BaseModel):
    status: str
    environment: str
    embedding_provider: str
    model: str
    speed: str
    features: list[str]
    documents_loaded: int

# ─────────────────────────────────────────────
# Streaming Generator
# ─────────────────────────────────────────────
def stream_answer(question: str, session_id: str) -> Generator[str, None, None]:
    """Stream answer with full conversation history"""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    
    config = {"configurable": {"session_id": session_id}}
    input_dict = {"question": question}
    
    try:
        for chunk in rag_chain_with_history.stream(input_dict, config=config):
            yield chunk
        
        history = session_store[session_id]
        print(f"✓ Stream completed for session {session_id[:8]}... ({len(history.messages)} messages)")
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"✗ Stream error: {error_msg}")
        yield error_msg

# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    try:
        collection = vectorstore._collection
        doc_count = collection.count()
    except:
        doc_count = 0
    
    return HealthResponse(
        status="healthy",
        environment="railway" if IS_RAILWAY else "local",
        embedding_provider="Cohere API (embed-english-light-v3.0)",
        model="llama-3.3-70b-versatile (via Groq)",
        speed="Fast (2-5 seconds)",
        features=["RAG", "Session Memory", "Streaming", "FAQ Support", "Lightweight (<1GB)"],
        documents_loaded=doc_count
    )

@app.post("/session/new", response_model=SessionResponse)
async def create_session():
    """Create a new chat session"""
    session_id = str(uuid4())
    session_store[session_id] = ChatMessageHistory()
    print(f"✓ New session created: {session_id[:8]}...")
    return SessionResponse(session_id=session_id)

@app.post("/ask")
async def ask(req: QuestionRequest):
    """Ask a question with full conversation history"""
    session_id = req.session_id or str(uuid4())
    
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    
    config = {"configurable": {"session_id": session_id}}
    input_dict = {"question": req.question}
    
    try:
        print(f"💬 Processing: '{req.question[:50]}...' for session {session_id[:8]}...")
        answer = rag_chain_with_history.invoke(input_dict, config=config)
        
        history = session_store[session_id]
        print(f"✓ Answer generated ({len(history.messages)} messages in history)")
        
        return {
            "answer": answer,
            "session_id": session_id,
            "message_count": len(history.messages)
        }
    except Exception as e:
        print(f"✗ Error in ask endpoint: {e}")
        raise

@app.post("/ask/stream")
async def ask_stream(req: QuestionRequest):
    """Stream answer with full conversation history"""
    session_id = req.session_id or str(uuid4())
    
    print(f"🌊 Starting stream for session {session_id[:8]}...")
    
    return StreamingResponse(
        stream_answer(req.question, session_id),
        media_type="text/plain",
        headers={
            "X-Session-ID": session_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific chat session"""
    if session_id in session_store:
        del session_store[session_id]
        print(f"✓ Session cleared: {session_id[:8]}...")
        return {"status": "Session cleared", "session_id": session_id}
    return {"status": "Session not found", "session_id": session_id}

@app.get("/session/{session_id}/history")
async def get_session_history_endpoint(session_id: str):
    """Get the chat history for a session"""
    if session_id in session_store:
        history = session_store[session_id]
        messages = [
            {
                "type": msg.type,
                "content": msg.content
            }
            for msg in history.messages
        ]
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages
        }
    return {"status": "Session not found", "session_id": session_id}

@app.get("/stats")
async def get_stats():
    """Get chatbot statistics"""
    try:
        collection = vectorstore._collection
        doc_count = collection.count()
    except:
        doc_count = 0
    
    return {
        "environment": "railway" if IS_RAILWAY else "local",
        "embedding_provider": "Cohere API",
        "active_sessions": len(session_store),
        "total_chunks": doc_count,
        "retriever_config": {
            "search_type": "mmr",
            "k": 6,
            "fetch_k": 15,
            "lambda_mult": 0.7
        },
        "documents": {
            "resume": os.path.exists(RESUME_MD_PATH),
            "faq": os.path.exists(FAQ_PATH),
        }
    }

# ─────────────────────────────────────────────
# Startup Event
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Print startup information"""
    print("\n" + "="*60)
    print("🚀 Portfolio Chatbot API Started Successfully!")
    print("="*60)
    print(f"🌍 Environment: {'Railway' if IS_RAILWAY else 'Local'}")
    print(f"🔌 Port: {PORT}")
    print(f"📦 Embeddings: Cohere API (lightweight)")
    print(f"📁 Vector Store: {VECTOR_DB_DIR}")
    print(f"📄 Resume: {RESUME_MD_PATH} {'✓' if os.path.exists(RESUME_MD_PATH) else '✗'}")
    print(f"❓ FAQ: {FAQ_PATH} {'✓' if os.path.exists(FAQ_PATH) else '✗'}")
    print(f"🤖 Model: llama-3.3-70b-versatile (Groq)")
    print(f"⚡ Speed: 2-5 seconds per response")
    print(f"💾 Image Size: < 1 GB (Railway optimized)")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)