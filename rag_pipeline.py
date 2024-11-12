from typing import Optional, Dict, Any, List
import os
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time
from dataclasses import dataclass
from loguru import logger
import json
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import asyncio
from datetime import datetime

@dataclass
class RAGConfig:
    """Configuration for the RAG Pipeline"""
    api_key: str
    model: str = "llama2"  # Default model
    temperature: float = 0.1
    max_retries: int = 3
    timeout: int = 30
    scraper_url: str = "http://localhost:8000/scrape"
    use_local_llm: bool = True  # Flag to determine whether to use local LLM
    local_llm_base_url: str = "http://localhost:11434/v1"  # Ollama API endpoint
    system_prompts: Dict[str, str] = None

    def __post_init__(self):
        self.system_prompts = {
            'query_generation': (
                "You are a search query optimizer. Take user input and return a search query "
                "that would get the most relevant results from the internet. "
                "Return only the optimized search query without any explanation or additional text."
                f"always refer to the latest date if needed, that is {datetime.now()}"
            ),
            'summarization': (
                "You are an expert summarizer. Provide a comprehensive summary of the given text "
                "while retaining all important information, facts, and key points. "
                "Focus on accuracy and completeness while being concise."
            ),
            'final_answer': (
                "You are a question answering expert. Based ONLY on the provided context, "
                "answer the original question accurately and comprehensively. "
                "If the context doesn't contain enough information to fully answer the question, "
                "acknowledge this in your response. Cite specific sources when possible."
            )
        }

class RAGPipeline:
    """Production-grade RAG Pipeline"""

    def __init__(self, config: Optional[RAGConfig] = None):
        if config is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            self.config = RAGConfig(api_key=api_key)
        else:
            self.config = config

        # Initialize client based on configuration
        if self.config.use_local_llm:
            self.client = OpenAI(
                base_url=self.config.local_llm_base_url,
                api_key="ollama",  # Required but not used by Ollama
                timeout=self.config.timeout
            )
        else:
            self.client = OpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )

        logger.info(f"Initialized RAG Pipeline with model: {self.config.model} "
                   f"using {'local LLM' if self.config.use_local_llm else 'OpenAI API'}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_search_query(self, user_input: str) -> Dict[str, Any]:
        """Generate optimized search query"""
        try:
            completion = self.client.chat.completions.create(
                model =  self.config.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompts['query_generation']},
                    {"role": "user", "content": user_input}
                ],
                temperature=self.config.temperature
            )
            return {'success': True, 'query': completion.choices[0].message.content.strip()}
        except Exception as e:
            logger.error(f"Query generation failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search_duckduckgo(self, query: str) -> List[str]:
        """Search DuckDuckGo and return top 5 URLs"""
        encoded_query = quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            ) as response:
                if response.status != 200:
                    raise Exception(f"Search failed with status {response.status}")
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                urls = []
                for result in soup.select('.result__url'):
                    href = result.get('href')
                    if href and len(urls) < 2: # take top N number of URLs
                        urls.append(href)
                
                return urls[:5]

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape single URL using scraper API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.scraper_url,
                    json={"url": url, "formats": ["markdown"]}
                ) as response:
                    result = await response.json()
                    return {
                        'success': True,
                        'url': url,
                        'content': result.get('data', {}).get('markdown', '')
                    }
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {str(e)}")
            return {'success': False, 'url': url, 'error': str(e)}

    async def scrape_all_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently"""
        tasks = [self.scrape_url(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def summarize_content(self, content: str) -> str:
        """Summarize content using OpenAI"""
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompts['summarization']},
                    {"role": "user", "content": content}
                ],
                temperature=self.config.temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return ""

    async def generate_final_answer(self, question: str, summaries: List[Dict[str, Any]]) -> str:
        """Generate final answer based on summaries"""
        try:
            context = "\n\n".join([
                f"Source {i+1} ({summary['url']}):\n{summary['summary']}"
                for i, summary in enumerate(summaries)
            ])

            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompts['final_answer']},
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
                ],
                temperature=self.config.temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "Failed to generate answer due to an error."

    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Complete RAG pipeline processing"""
        start_time = time.time()
        
        try:
            # Step 1: Generate optimized search query
            logger.info("Generating search query...")
            query_result = await self.generate_search_query(user_query)
            if not query_result['success']:
                return {'success': False, 'error': query_result['error']}
            
            logger.info(f"Generated query {query_result['query']}")

            # Step 2: Get URLs from DuckDuckGo
            logger.info("Searching DuckDuckGo...")
            # urls = await self.search_duckduckgo(query_result['query'])
            urls = await self.search_duckduckgo(user_query)
            logger.info(f"Found {len(urls)} URLs")

            # Step 3: Scrape all URLs in parallel
            logger.info("Scraping URLs...")
            scraped_contents = await self.scrape_all_urls(urls)
            
            # Step 4: Summarize successful scrapes
            summaries = []
            for content in scraped_contents:
                if content['success']:
                    logger.info(f"Summarizing content from {content['url']}...")
                    summary = await self.summarize_content(content['content'])
                    summaries.append({
                        'url': content['url'],
                        'summary': summary
                    })

            # Step 5: Generate final answer
            logger.info("Generating final answer...")
            final_answer = await self.generate_final_answer(user_query, summaries)

            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'original_query': user_query,
                'optimized_query': query_result['query'],
                'urls': urls,
                'summaries': summaries,
                'answer': final_answer,
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

async def main():
    """Main function to demonstrate usage"""
    logger.remove()
    logger.add(
        "search.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    logger.add(lambda msg: print(msg), level="INFO")

    try:
        pipeline = RAGPipeline()
        user_question = input("Enter your question: ")
        
        logger.info(f"\nProcessing question: {user_question}")
        result = await pipeline.process_query(user_question)
        
        if result['success']:
            print("\nSearch Results:")
            print(f"Original Query: {result['original_query']}")
            print(f"Optimized Query: {result['optimized_query']}")
            
            print("\nFound URLs:")
            for i, url in enumerate(result['urls'], 1):
                print(f"{i}. {url}")
            
            print("\nSummaries:")
            for i, summary in enumerate(result['summaries'], 1):
                print(f"\nSource {i}: {summary['url']}")
                print(f"Summary: {summary['summary'][:200]}...")
            
            print("\nFinal Answer:")
            print(result['answer'])
            
            print(f"\nTotal Processing Time: {result['processing_time']:.2f} seconds")
        else:
            print(f"\nError: {result['error']}")
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())