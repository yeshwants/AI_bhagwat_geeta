import os
from dotenv import load_dotenv
from agent.agent import ask_gita
from agent.memory import clear_session, get_session_history

load_dotenv()

# ─── Configuration ────────────────────────────────────────
SESSION_ID = "user_session"

WELCOME_MESSAGE = """
╔══════════════════════════════════════════════════════════╗
║         🕉️  BHAGAVAD GITA AI ASSISTANT  🕉️              ║
║                                                          ║
║  Ask any question about the Gita's teachings.            ║
║  Type 'clear' to start a fresh conversation.             ║
║  Type 'history' to see conversation so far.              ║
║  Type 'quit' or 'exit' to leave.                         ║
╚══════════════════════════════════════════════════════════╝
"""

DIVIDER = "─" * 60


def show_history():
    """Prints the current conversation history."""
    messages = get_session_history(SESSION_ID).messages
    if not messages:
        print("\n  (No conversation history yet)\n")
        return

    print(f"\n📜 Conversation History ({len(messages)} messages):")
    print(DIVIDER)
    for msg in messages:
        if msg.type == "human":
            print(f"\n🧑 You   : {msg.content}")
        else:
            print(f"\n🤖 Gita  : {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
    print()


def handle_special_commands(user_input: str) -> bool:
    """
    Handles special commands. Returns True if a command was handled,
    False if the input should be treated as a normal question.
    """
    command = user_input.strip().lower()

    if command in ("quit", "exit"):
        print("\n🙏 Hari Om. May the Gita's wisdom guide your path.\n")
        exit(0)

    if command == "clear":
        clear_session(SESSION_ID)
        print("✨ Conversation cleared. Starting fresh.\n")
        return True

    if command == "history":
        show_history()
        return True

    if command == "help":
        print("""
Commands:
  clear    → Start a new conversation
  history  → View conversation so far
  quit     → Exit the assistant
  help     → Show this message
        """)
        return True

    return False  # not a command — treat as a question


def main():
    print(WELCOME_MESSAGE)

    while True:
        try:
            # Get user input
            user_input = input("🧑 You   : ").strip()

            # Skip empty input
            if not user_input:
                continue

            # Handle special commands
            if handle_special_commands(user_input):
                continue

            # Get answer from the Gita assistant
            print(f"\n🤖 Gita  : ", end="", flush=True)
            answer = ask_gita(user_input, SESSION_ID)
            print(answer)
            print(f"\n{DIVIDER}")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n🙏 Hari Om. May the Gita's wisdom guide your path.\n")
            break

        except Exception as e:
            print(f"\n⚠️  Something went wrong: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()