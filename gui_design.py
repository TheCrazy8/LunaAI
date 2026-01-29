"""
GUI Design Documentation for LunaAI Chat Interface
This module provides a visual representation of the chat GUI design.
"""

def print_gui_layout():
    """Print ASCII representation of the GUI layout."""
    print("="*80)
    print("LunaAI Chat GUI - Interface Design (sv_ttk themed)")
    print("="*80)
    print()
    
    gui_layout = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🌙 LunaAI Chat                                        Model ready ✓         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [SYSTEM] Model loaded successfully! You can start chatting.               │
│                                                                             │
│  You: Hello! How are you today?                                            │
│                                                                             │
│  LunaAI: Hello! I'm doing well, thank you for asking. I'm here to         │
│  help you with any questions or conversations you'd like to have.          │
│  How can I assist you today?                                               │
│                                                                             │
│  You: Can you help me with a coding problem?                               │
│                                                                             │
│  LunaAI: Of course! I'd be happy to help you with your coding problem.     │
│  Please describe the issue you're facing, and I'll do my best to           │
│  provide assistance.                                                        │
│                                                                             │
│  [Chat display area with scrollbar]                                        │
│                                                                             │
│                                                                             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ [Type your message here...                                    ] [ Send ]   │
├─────────────────────────────────────────────────────────────────────────────┤
│            [ Clear Chat ]  [ Toggle Theme ]  [ Export Chat ]               │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    print(gui_layout)
    print()


def print_features():
    """Print GUI features."""
    print("="*80)
    print("GUI Features")
    print("="*80)
    print()
    
    features = [
        ("Title Bar", "Shows 'LunaAI Chat' and model status indicator"),
        ("Status Label", "Displays current status (Loading/Ready/Generating)"),
        ("Chat Display", "Scrollable text area with color-coded messages:"),
        ("  - User messages", "Green (#4CAF50)"),
        ("  - Assistant messages", "Blue (#2196F3)"),
        ("  - System messages", "Orange (#FFA500), italic"),
        ("Input Field", "Text entry for user messages (Enter to send)"),
        ("Send Button", "Sends the message (disabled during generation)"),
        ("Clear Chat", "Clears conversation history with confirmation"),
        ("Toggle Theme", "Switches between dark and light sv_ttk themes"),
        ("Export Chat", "Saves conversation history to text file"),
    ]
    
    for feature, description in features:
        print(f"  • {feature:20} : {description}")
    print()


def print_theme_info():
    """Print theme information."""
    print("="*80)
    print("Theme Support (sv_ttk)")
    print("="*80)
    print()
    
    print("Dark Theme (Default):")
    print("  - Background: #1e1e1e")
    print("  - Foreground: #ffffff")
    print("  - Modern, eye-friendly appearance")
    print()
    
    print("Light Theme:")
    print("  - Background: #ffffff")
    print("  - Foreground: #000000")
    print("  - Clean, professional appearance")
    print()
    
    print("Toggle between themes with 'Toggle Theme' button")
    print("Theme applies to entire window and chat display")
    print()


def print_interaction_flow():
    """Print user interaction flow."""
    print("="*80)
    print("User Interaction Flow")
    print("="*80)
    print()
    
    steps = [
        "1. Application Launch",
        "   → Window opens with 'Loading model...' message",
        "   → Model loads asynchronously in background thread",
        "",
        "2. Model Loading Complete",
        "   → Status changes to 'Model ready ✓'",
        "   → Send button becomes enabled",
        "   → Input field becomes active and focused",
        "",
        "3. User Input",
        "   → Type message in input field",
        "   → Press Enter or click Send button",
        "   → Message appears in chat display (green)",
        "",
        "4. AI Response Generation",
        "   → Status changes to 'Generating response...'",
        "   → Input and Send button disabled temporarily",
        "   → Model generates response in background thread",
        "",
        "5. Response Display",
        "   → AI response appears in chat display (blue)",
        "   → Status returns to 'Model ready ✓'",
        "   → Input and Send button re-enabled",
        "   → Ready for next message",
        "",
        "6. Additional Actions",
        "   → Clear Chat: Clears all messages with confirmation",
        "   → Toggle Theme: Switches UI appearance",
        "   → Export Chat: Saves conversation to text file",
    ]
    
    for step in steps:
        print(step)
    print()


def main():
    """Main function to display all GUI documentation."""
    print_gui_layout()
    print_features()
    print_theme_info()
    print_interaction_flow()
    
    print("="*80)
    print("Technical Implementation")
    print("="*80)
    print()
    print("Framework: Tkinter with TTK widgets")
    print("Theme: sv_ttk (Sun Valley theme)")
    print("Threading: Async model loading and response generation")
    print("Font: Helvetica (11pt for messages, 18pt bold for title)")
    print("Layout: Grid-based responsive design")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
