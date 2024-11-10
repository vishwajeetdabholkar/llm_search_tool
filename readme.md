# LLM Search Tool 🔍

A powerful LLM search assistant that combines web search, content analysis, and AI-powered summarization to provide comprehensive answers to user queries. Built with Streamlit, OpenAI/Ollama, and modern async Python.

## 🌟 Features

- **Flexible LLM Support**: Use either OpenAI's models or local LLMs through Ollama
- **Advanced Search Pipeline**: 
  - Query optimization
  - Web search through DuckDuckGo
  - Parallel content scraping
  - AI-powered summarization
  - Contextual answer generation
- **Interactive UI**: Clean, responsive Streamlit interface with real-time progress tracking
- **Search History**: Keep track of past queries and results
- **Configurable Settings**: Adjust model parameters, temperature, and other settings on the fly

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Language Models**: 
  - OpenAI API (GPT-4, GPT-4-mini)
  - Local LLMs via Ollama (Llama 3.1, 3.2)
- **Web Scraping**: aiohttp, BeautifulSoup4
- **Async Processing**: Python asyncio
- **Error Handling**: Tenacity for retries
- **Logging**: Loguru

## 📋 Prerequisites

- Python 3.8+
- An OpenAI API key (if using OpenAI models)
- Ollama setup (if using local LLMs)
- A running scraper service

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag_pipeline_search_tool.git
cd rag_pipeline_search_tool
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` file with your configuration:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_RETRIES=100
OPENAI_TIMEOUT=30
```

## 🎮 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Configure your preferences in the sidebar:
   - Choose between OpenAI or Local LLM
   - Select model
   - Adjust temperature and other parameters
   - Enter API keys if required

4. Enter your query and click "Search"

## 🔧 Configuration Options

### LLM Settings
- **Use Local LLM**: Toggle between OpenAI and local LLM
- **Model Selection**: 
  - OpenAI: gpt-4o-mini, gpt-4
  - Local: llama3.1, llama3.2
- **Temperature**: Controls response randomness (0.0 - 1.0)

### Pipeline Configuration
- **Max Retries**: Number of retry attempts for failed requests
- **Timeout**: Request timeout in seconds
- **Scraper URL**: Endpoint for web scraping service

## 📊 Pipeline Process

1. **Query Optimization**
   - Input query is processed to create an optimized search query
   - Uses LLM to improve search relevance

2. **Web Search**
   - Searches DuckDuckGo for relevant sources
   - Returns top 5 most relevant URLs

3. **Content Retrieval**
   - Parallel scraping of all URLs
   - Content extraction and cleaning

4. **Summarization**
   - Each source is summarized independently
   - Maintains key information and facts

5. **Answer Generation**
   - Combines all summaries and context
   - Generates comprehensive, cited answer

## 🔍 Search History

The tool maintains a session history of:
- Timestamps
- Original queries
- Processing times
- Number of sources
- Complete results

## ⚠️ Error Handling

- Robust retry mechanism for API calls
- Graceful failure handling
- Detailed error logging
- User-friendly error messages

## 📝 Logging

Comprehensive logging system using Loguru:
- File-based logging
- Console output
- Timestamp and level-based logging
- Error tracking

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Vishwajeet

## 🙏 Acknowledgments

- OpenAI for their API
- Ollama for local LLM support
- Streamlit for the wonderful UI framework
- DuckDuckGo for search capabilities

## 🐛 Known Issues & Limitations

- DuckDuckGo may rate-limit extensive searches
- Local LLM performance depends on hardware capabilities
- Requires active internet connection for web search
- Limited to 5 sources per query for performance

## 🔜 Future Improvements

- Add support for more search engines
- Implement caching for frequently asked queries
- Add export functionality for search results
- Enhance source filtering and ranking
- Add support for different language models


## Screenshots
![image](https://github.com/user-attachments/assets/eec78146-92d6-4baa-b884-0afd0cf4a4c2)
![image](https://github.com/user-attachments/assets/953b3a08-4d1e-466f-a0cd-a9dd0d09bc29)
![image](https://github.com/user-attachments/assets/e713b118-2b3a-408c-b6a1-ad87a2f5bd5f)
![image](https://github.com/user-attachments/assets/121d580e-119b-454b-8390-3b492f7088ff)


