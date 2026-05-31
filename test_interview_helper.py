#!/usr/bin/env python3
"""Test script for InterviewHelper RAG system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from interview_helper.core import InterviewHelper


def test_basic_workflow():
    """Test the complete RAG workflow."""
    print("\n" + "="*60)
    print("TESTING: InterviewHelper Basic Workflow")
    print("="*60)

    # Initialize helper
    print("\n1. Initializing InterviewHelper...")
    try:
        helper = InterviewHelper()
        print("   ✓ InterviewHelper initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return False

    # Test interview content
    interview_content = """
I recently had a technical interview at Google for the Software Engineer position.
The interview was 60 minutes long and consisted of two parts:

1. Coding Problem (45 minutes)
   - Write a function to find all subarrays with sum equal to k
   - I solved it using a sliding window approach in O(n) time complexity
   - I handled edge cases like negative numbers and empty arrays
   - The interviewer appreciated the efficient solution

2. System Design (15 minutes)
   - We discussed how to design a URL shortener service
   - I talked about caching strategies, database sharding, and API design
   - Asked good questions about team structure and tech stack

Overall, I felt positive about my performance. The interviewer was encouraging
and took time to explain concepts clearly. I'm confident this will lead to an offer.
    """.strip()

    metadata = {
        "company": "Google",
        "role": "Software Engineer",
        "date": "2026-05-30",
        "location": "San Francisco, CA",
        "tags": ["technical", "coding", "google"],
        "interview_type": "Technical Round 1"
    }

    # Add interview
    print(f"\n2. Adding interview (company: {metadata['company']}...)")
    try:
        doc_ids = helper.add_interview(
            content=interview_content,
            metadata=metadata,
            chunk_size=500
        )
        print(f"   ✓ Added {len(doc_ids)} document chunks to vector store")
    except Exception as e:
        print(f"   ✗ Failed to add interview: {e}")
        return False

    # List all interviews
    print("\n3. Listing all interviews...")
    try:
        all_interviews = helper.get_all_interviews()
        print(f"   ✓ Found {len(all_interviews)} interview(s) in database")
        for i, interview in enumerate(all_interviews[:3], 1):
            meta = interview.get('metadata', {})
            print(f"      - Interview #{i}: {meta.get('company', 'Unknown')} | "
                  f"{meta.get('role', 'N/A')} | {meta.get('date', 'N/A')}")
    except Exception as e:
        print(f"   ✗ Failed to list interviews: {e}")
        return False

    # Search for content
    print("\n4. Searching for 'coding problem solution'...")
    try:
        search_results = helper.search("coding problem solution", top_k=3)
        print(f"   ✓ Found {len(search_results.documents)} relevant chunks")
        if search_results.documents:
            meta = search_results.documents[0].get('metadata', {})
            print(f"      Context from: {meta.get('company', 'Unknown')} "
                  f"(first 200 chars): {search_results.documents[0]['text'][:200]}...")
    except Exception as e:
        print(f"   ✗ Search failed: {e}")
        return False

    # Ask question (this tests the LLM generation)
    print("\n5. Asking: 'How did I perform on the coding question?'...")
    try:
        answer = helper.ask("How did I perform on the coding question?")
        print(f"   ✓ Got response ({len(answer)} chars)")
        if len(answer) < 50:
            print(f"      Response seems short: {answer}")
        else:
            print(f"      Answer preview: {answer[:200]}...")
    except Exception as e:
        print(f"   ✗ Failed to ask question: {e}")
        return False

    # Clean up - clear the vector store
    print("\n6. Cleaning up test data...")
    try:
        from interview_helper.vector_store import VectorStore
        collection = VectorStore()
        collection.client.delete_all_collections()
        print("   ✓ Cleared ChromaDB collections")
    except Exception as e:
        print(f"   Note - cleanup issue (non-critical): {e}")

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_basic_workflow()
    sys.exit(0 if success else 1)
