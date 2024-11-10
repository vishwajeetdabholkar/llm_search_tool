import streamlit as st
import asyncio
from typing import Optional, Dict, Any, List
import os
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from dataclasses import dataclass
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import json
from datetime import datetime

# Import the RAGPipeline class from previous file
from rag_pipeline import RAGPipeline, RAGConfig

# Streamlit page config
st.set_page_config(
    page_title="RAG Search Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTextInput>div>div>input {
        padding: 0.5rem 1rem;
    }
    .stProgress>div>div>div {
        height: 10px;
    }
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metrics-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session states
if 'history' not in st.session_state:
    st.session_state.history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
if 'use_local_llm' not in st.session_state:
    st.session_state.use_local_llm = True
if 'model' not in st.session_state:
    st.session_state.model = "llama3.2"
if 'local_url' not in st.session_state:
    st.session_state.local_url = "http://localhost:11434/v1"
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # LLM Selection
    st.session_state.use_local_llm = st.checkbox("Use Local LLM", value=st.session_state.use_local_llm)
    
    if st.session_state.use_local_llm:
        st.session_state.model = st.selectbox(
            "Model",
            ["llama3.2", "llama3.1"],
            index=0
        )
        st.session_state.local_url = st.text_input(
            "Local LLM URL",
            value=st.session_state.local_url
        )
    else:
        # API Key input
        st.session_state.api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your OpenAI API key"
        )
        st.session_state.model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4"],
            index=0
        )
    
    # Advanced settings
    st.subheader("Advanced Settings")
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Controls randomness in the response"
    )

# Main content
st.title("🔍 RAG Search Assistant")
st.markdown("""
    Get comprehensive answers to your questions using our RAG (Retrieval Augmented Generation) pipeline.
    This tool searches the internet, analyzes multiple sources, and provides detailed answers with citations.
""")

# Query input
query = st.text_input(
    "Enter your question",
    placeholder="e.g., What are the latest developments in quantum computing?"
)

# Process button
if st.button(
    "Search",
    type="primary",
    disabled=not (query and (st.session_state.use_local_llm or st.session_state.api_key))
):
    try:
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize RAG Pipeline
        config = RAGConfig(
            api_key=st.session_state.api_key if not st.session_state.use_local_llm else "ollama",
            model=st.session_state.model,
            temperature=st.session_state.temperature,
            use_local_llm=st.session_state.use_local_llm,
            local_llm_base_url=st.session_state.local_url if st.session_state.use_local_llm else None
        )
        pipeline = RAGPipeline(config)
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='metrics-box'>🔍 Search Phase</div>", unsafe_allow_html=True)
            search_status = st.empty()
        with col2:
            st.markdown("<div class='metrics-box'>📝 Analysis Phase</div>", unsafe_allow_html=True)
            analysis_status = st.empty()
        with col3:
            st.markdown("<div class='metrics-box'>⏱️ Processing Time</div>", unsafe_allow_html=True)
            time_status = st.empty()

        # Process the query
        status_text.text("🔍 Initializing search...")
        progress_bar.progress(10)
        
        # Run the pipeline
        result = asyncio.run(pipeline.process_query(query))
        
        if result['success']:
            # Update progress
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Display results in an organized way
            st.markdown("### 📊 Search Results")
            
            # Query information
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown("**Original Query:**")
            st.write(result['original_query'])
            st.markdown("**Optimized Search Query:**")
            st.write(result['optimized_query'])
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Sources and summaries
            st.markdown("### 📚 Sources and Summaries")
            for i, (url, summary) in enumerate(zip(result['urls'], result['summaries']), 1):
                st.markdown(f"<div class='source-box'>", unsafe_allow_html=True)
                st.markdown(f"**Source {i}**: [{url}]({url})")
                with st.expander("View Summary"):
                    st.write(summary['summary'])
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Final answer
            st.markdown("### 🎯 Final Answer")
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.write(result['answer'])
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Save to history
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'query': query,
                'result': result
            })
            
            # Update metrics
            search_status.success(f"{len(result['urls'])} sources found")
            analysis_status.success(f"{len(result['summaries'])} sources analyzed")
            time_status.info(f"{result['processing_time']:.2f} seconds")
            
        else:
            st.error(f"Error: {result['error']}")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
# History section in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📜 Search History")
    for item in reversed(st.session_state.history):
        with st.expander(f"{item['timestamp']} - {item['query'][:50]}..."):
            st.write(f"Query: {item['query']}")
            st.write(f"Processing Time: {item['result']['processing_time']:.2f}s")
            st.write(f"Sources: {len(item['result']['urls'])}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and LLMs</p>
        <p>By Vishwajeet</p>
    </div>
    """,
    unsafe_allow_html=True
)