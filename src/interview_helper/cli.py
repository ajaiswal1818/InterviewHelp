"""CLI interface for Interview Helper using Rich."""

import logging
from typing import Optional
import click
from pathlib import Path

# Initialize helper and add interview
from .core import InterviewHelper
from .llm import create_llm_client, available_providers


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


def _build_helper(provider=None, model=None):
    """Build an InterviewHelper backed by the selected local LLM provider."""
    client = create_llm_client(provider=provider, model=model)
    return InterviewHelper(llm_client=client)


def provider_options(f):
    """Shared --provider/--model options for commands that talk to the LLM."""
    f = click.option(
        '--model', '-m', default=None,
        help='Model name (defaults to the provider default or LLM_MODEL env)'
    )(f)
    f = click.option(
        '--provider', '-p', default=None,
        type=click.Choice(available_providers(), case_sensitive=False),
        help='Local LLM provider (default: LLM_PROVIDER env, else mlx)'
    )(f)
    return f


@click.group(invoke_without_command=True)
@click.pass_context
def interview_helper(ctx):
        """Interview Helper - Personal Interview RAG System (ChromaDB + pluggable local LLM)."""
        if ctx.invoked_subcommand is None:
            click.echo(ctx.get_help())


@interview_helper.command()
@click.option('--content', '-c', default=None, help='Text content to add')
@click.option('--file', '-f', 'file_path',
              type=click.Path(exists=True, dir_okay=False),
              default=None, help='Path to a .txt/.md/.pdf file to ingest')
@click.option('--dir', 'dir_path',
              type=click.Path(exists=True, file_okay=False),
              default=None, help='Directory of documents to ingest (recursive)')
@click.option('--title', '-t', default=None, help='Title (defaults to company/role or filename)')
@click.option('--company', default=None, help='Company name')
@click.option('--role', default=None, help='Role applied for')
@click.option('--date', '-d', default="2026-05-29", help='Date of interview (YYYY-MM-DD)')
@click.option('--location', default=None, help='Interview location')
@click.option('--tags', '-tag', multiple=True, default=(), help='Tags for the document')
@provider_options
def add(content, file_path, dir_path, title, company, role, date, location, tags, provider, model):
    """Add content to your database from text (--content), a file (--file), or a folder (--dir)."""
    # Exactly one input source is required
    sources = [s for s in (content, file_path, dir_path) if s]
    if len(sources) != 1:
        raise click.UsageError("Provide exactly one of --content, --file, or --dir.")

    tag_list = list(tags)
    base_metadata = {
        "date": date,
        "company": company or "",
        "role": role or "",
        "location": location or "",
        "tags": tag_list if tag_list else ["interview"],
    }

    try:
        helper = _build_helper(provider, model)

        if dir_path:
            logger.info(f"Ingesting directory: {dir_path}")
            results = helper.add_directory(dir_path, metadata=base_metadata)
            total = sum(len(v) for v in results.values())
            print(f"\n{Color.GREEN}Ingested {len(results)} file(s), "
                  f"{total} chunks from {dir_path}{Color.RESET}")
        elif file_path:
            logger.info(f"Ingesting file: {file_path}")
            added = helper.add_document(file_path, metadata=base_metadata)
            print(f"\n{Color.GREEN}Added {len(added)} chunks from {file_path}{Color.RESET}")
        else:
            if not title:
                title = " - ".join([p for p in (company, role) if p]) or "Untitled Interview"
            logger.info(f"Adding interview with title: {title}")
            metadata = {"title": title, **base_metadata}
            added = helper.add_interview(content, metadata)
            print(f"\n{Color.GREEN}Successfully added {len(added)} chunks "
                  f"to your interview database!{Color.RESET}")
    except Exception as e:
        print(f"\n{Color.RED}Error adding content: {e}{Color.RESET}")
        raise


@interview_helper.command()
@click.argument('query', nargs=-1, required=True)
@provider_options
def search(query, provider, model):
    """Search through your interviews."""
    logger.info(f"Searching for: {' '.join(query)}")

    try:
        helper = _build_helper(provider, model)

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
                tags = metadata["tags"]
                tag_str = tags if isinstance(tags, str) else ", ".join(tags)
                print(f"   Tags: {Color.MAGENTA}{tag_str}{Color.RESET}")

        print(f"\n{Color.CYAN}--- Use 'interview-helper ask' to get detailed answers about these topics.{Color.RESET}\n")
    except Exception as e:
        print(f"\n{Color.RED}Search error: {e}{Color.RESET}")
        raise


@interview_helper.command()
@click.argument('question', nargs=-1, required=True)
@click.option('--company', default=None, help='Filter context by company name')
@provider_options
def ask(question, company, provider, model):
    """Ask questions about your interviews."""
    logger.info(f"Asking question: {' '.join(question)}")

    try:
        helper = _build_helper(provider, model)
        client = helper.llm_client

        where_filter = {"company": company} if company else None
        results = helper.search(" ".join(question), where_clause=where_filter)
        
        if not results.documents:
            print(f"\n{Color.YELLOW}No relevant context found for: {' '.join(question)}")
            return

        # Prepare prompt with retrieved context
        system_prompt = (
            "You are an interview helper assistant. "
            "Answer questions based on the context provided from the user's interviews. "
            "Be concise, helpful, and cite specific details from the context."
        )

        context = results.documents[0].get('text', '') if results.documents else 'No context available'
        user_prompt = f"""{system_prompt}

Context retrieved from interview database:

{context[:1000]}

Question: {' '.join(question)}"""

        print(f"\n{Color.CYAN}Generating answer... (this may take a few seconds){Color.RESET}")
        logger.info(f"User prompt for LLM:\n{user_prompt}")
        # Generate response (model=None falls back to the provider default)
        result = client.generate(prompt=user_prompt, model=model)
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


@interview_helper.command(name="list")
def list_all():
    """List all interviews in your database."""
    logger.info("Listing all interviews")

    try:
        helper = _build_helper()

        interviews = helper.get_all_interviews()
        
        if not interviews:
            print(f"\n{Color.YELLOW}Your database is empty!{Color.RESET}")
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
            print(f"   Role: {role if role else 'N/A'}")
            print(f"   Date: {date}")

            tags = metadata.get('tags', [])
            if tags:
                tag_str = tags if isinstance(tags, str) else ", ".join(tags)
                print(f"   Tags: {Color.MAGENTA}{tag_str}{Color.RESET}")

            print("")
    except Exception as e:
        print(f"\n{Color.RED}Error listing interviews: {e}{Color.RESET}")
        raise


@interview_helper.command(name="clear")
@click.option('--yes', '-y', is_flag=True, default=False, help='Skip the confirmation prompt')
def clear(yes):
    """Delete all interviews from your database."""
    logger.info("Clearing all interviews")

    if not yes:
        confirm = click.confirm(
            "This will permanently delete ALL stored interviews. Continue?",
            default=False,
        )
        if not confirm:
            print(f"\n{Color.YELLOW}Aborted. Nothing was deleted.{Color.RESET}")
            return

    try:
        helper = _build_helper()
        helper.clear_all_interviews()
        print(f"\n{Color.GREEN}Cleared all interviews from your database.{Color.RESET}")
    except Exception as e:
        print(f"\n{Color.RED}Error clearing interviews: {e}{Color.RESET}")
        raise


if __name__ == "__main__":
    interview_helper()
