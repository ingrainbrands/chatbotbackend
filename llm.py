# llm.py — Iryax AI Assistant
# This file connects to Ollama and uses our RAG database for Iryax!

import ollama
import json
import sys
import os

# Add the project root to Python's sys.path so it can find the 'backend' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import the RAG pipeline we built
try:
    from backend.rag_pipeline import generate_rag_response
except ImportError as e:
    print(f"Error importing backend.rag_pipeline: {e}")
    print("Make sure your virtual environment (venv) is activated and dependencies are installed!")
    sys.exit(1)

def main():
    print("=" * 50)
    print("        Iryax AI Assistant (RAG Enabled)")
    print("=" * 50)
    print("Type your question and press Enter.")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit", "bye"):
            print("\nGoodbye! See you next time. 👋")
            break

        if not user_input:
            continue

        print("\nAssistant: ", end="", flush=True)
        
        # Get response using our RAG pipeline
        full_response = ""
        sources = []
        try:
            for chunk_str in generate_rag_response(user_input, history):
                chunk_data = json.loads(chunk_str)
                token = chunk_data.get("token", "").replace("\r", "")
                if token:
                    print(token, end="", flush=True)
                    full_response += token
                if chunk_data.get("sources"):
                    sources = chunk_data.get("sources")
        except Exception as e:
            print(f"\n[Error] {e}")
            

        print("\n")
        print("-" * 50)
        
        # Save to history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
