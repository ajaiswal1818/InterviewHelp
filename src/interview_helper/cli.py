"""CLI interface for Interview Helper using Rich."""

import logging
from typing import Optional
import click
from pathlib import Path

# Initialize helper and add interview
from .core import InterviewHelper
from .ollama_client import OllamaClient


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging():
    pass


class Color:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


@click.group(invoke_without_command=True)
def interview_helper():
        """Interview Helper - Personal Interview RAG System with ChromaDB and Ollama."""
        pass


@interview_helper.command()
@click.option('--content', '-c', required=True, prompt=False, help='Interview content')
@click.option('--title', '-t', required=True, prompt=False, help='Interview title')
@click.option('--company', default=None, help='Company name')
@click.option('--role', default=None, help='Role applied for')
@click.option('--date', '-d', default="2026-05-29", help='Date of interview (YYYY-MM-DD)')
@click.option('--location', default=None, help='Interview location')
@click.option('--tags', '-tag', multiple=True, default=(), help='Tags for the interview')
@click.option('--model', '-m', default="llama3", help='Ollama model to use')
def add(content, title, company, role, date, location, tags, model):
    """Add a new interview to your database."""
    logger.info(f"Adding interview with title: {title}")

    # Convert tags to list
    tag_list = list(tags)

    # Create metadata dictionary
    metadata = {
        "title": title,
        "date": date,
        "company": company or "",
        "role": role or "",
        "location": location or "",
        "tags": tag_list if tag_list else ["interview"],
    }

    try:
        client = OllamaClient()
        helper = InterviewHelper(ollama_client=client)
        
        added = helper.add_interview(content, metadata)
        print(f"\n{Color.GREEN}Successfully added {len(added)} chunks to your interview database!{Color.RESET}")
    except Exception as e:
        print(f"\n{Color.RED}Error adding interview: {e}{Color.RESET}")
        raise


@interview_helper.command()
@click.argument('query', nargs=-1, required=True)
@click.option('--model', '-m', default="llama3", help='Ollama model to use')
def search(query, model):
    """Search through your interviews."""
    logger.info(f"Searching for: {' '.join(query)}")

    try:
        from .core import InterviewHelper
        from .ollama_client import OllamaClient
        
        client = OllamaClient()
        helper = InterviewHelper(ollama_client=client)
        
        results = helper.search(" ".join(query))
        
        if not results.documents:
            print(f"\n{Color.YELLOW}No results found for: {' '.join(query)}{Color.RESET}")
            return
        
        # Display results
        print(f"\n{Color.GREEN}Found {len(results.documents)} relevant chunks:{Color.RESET}\n")
        
        for i, doc in enumerate(results.documents):
            text = doc.get("text", "")[:200] + ("..." if len(doc.get("text", "")) > 200 else "")
            metadata = doc.get("metadata", {})
            
            print(f"{Color.BLUE}📄 Result {i+1}{Color.RESET}")
            print(f"   From: {metadata.get('company', 'Unknown')} - {metadata.get('title', '')}")
            print(f"   Date: {metadata.get('date', 'Unknown')}")
            if "tags" in metadata:
                tags = metadata["tags"] if isinstance(metadata["tags"], list) else metadata.get("tags", [])
                tag_str = ", ".join(tags)
                print(f"   Tags: {Color.MAGENTA}{tag_str}{Color.RESET}")

        print(f"\n{Color.CYAN}--- Use 'interview-helper ask' to get detailed answers about these topics.{Color.RESET}\n")
    except Exception as e:
        print(f"\n{Color.RED}Search error: {e}{Color.RESET}")
        raise


@interview_helper.command()
@click.argument('question', nargs=-1, required=True)
def ask(question):
    """Ask questions about your interviews."""
    logger.info(f"Asking question: {' '.join(question)}")

    try:
        from .core import InterviewHelper
        from .ollama_client import OllamaClient
        
        client = OllamaClient()
        helper = InterviewHelper(ollama_client=client)
        
        results = helper.search(" ".join(question))
        
        if not results.documents:
            print(f"\n{Color.YELLOW}No relevant context found for: {' '.join(question)}")
            return

        # Prepare prompt with retrieved context
        system_prompt = (
            "You are an interview helper assistant. "
            "Answer questions based on the context provided from the user's interviews. "
            "Be concise, helpful, and cite specific details from the context."
        )

        user_prompt = f"""Context retrieved from interview database:

{results.documents[0].get('text', '')[:1000] if results.documents else 'No context available'}

Question: {' '.join(question)}"""

        print(f"\n{Color.CYAN}Generating answer... (this may take a few seconds){Color.RESET}")
        # ---------  Testing   -----------------
        logger.info(f"User prompt for LLM:\n{user_prompt}")
        return
        # ---------------------------------------
        # Generate response
        result = client.generate(prompt=user_prompt, model='qwen3.5:4b-mlx')
        answer = result.get("response", "No answer available")

        # Display answer
        print("\n" + "=" * 60 + "\n")
        print(f"{Color.GREEN}Answer:{Color.RESET}\n")
        for line in answer.split('\n'):
            if line.strip():
                print(f"    {line}")
        print("\n" + "=" * 60 + "\n")
    except Exception as e:
        print(f"\n{Color.RED}Error generating response: {e}{Color.RESET}")
        raise


@interview_helper.command()
def list_all():
    """List all interviews in your database."""
    logger.info("Listing all interviews")

    try:
        from .core import InterviewHelper
        from .ollama_client import OllamaClient
        
        client = OllamaClient()
        helper = InterviewHelper(ollama_client=client)
        
        interviews = helper.get_all_interviews()
        
        if not interviews:
            print("\n{Color.YELLOW}Your database is empty!{Color.RESET}")
            return

        # Display interviews sorted by date
        for i, interview in enumerate(interviews):
            metadata = interview.get("metadata", {})
            
            title = metadata.get('title', 'Untitled Interview')
            company = metadata.get('company', 'Unknown') or 'Unknown'
            date = metadata.get('date', 'Unknown Date')
            role = metadata.get('role', '') or ''

            print(f"{Color.GREEN}{i+1}. {title}{Color.RESET}")
            print(f"   Company: {Color.BLUE}{company}{Color.RESET}")
            print(f"   Role: {'{'.join(role) if role else 'N/A'}")
            print(f"   Date: {date}")

            tags = metadata.get('tags', [])
            if tags:
                tag_str = ", ".join(tags)
                print(f"   Tags: {Color.MAGENTA}{tag_str}{Color.RESET}")

            print("")
    except Exception as e:
        print(f"\n{Color.RED}Error listing interviews: {e}{Color.RESET}")
        raise


if __name__ == "__main__":
    interview_helper()
