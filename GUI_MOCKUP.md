# LunaAI Chat GUI Mockup

## Interface Design (sv_ttk themed)

```
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
├─────────────────────────────────────────────────────────────────────────────┤
│ [Type your message here...                                    ] [ Send ]   │
├─────────────────────────────────────────────────────────────────────────────┤
│            [ Clear Chat ]  [ Toggle Theme ]  [ Export Chat ]               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Color Scheme
- **User messages**: Green (#4CAF50) - Bold
- **Assistant messages**: Blue (#2196F3) - Bold
- **System messages**: Orange (#FFA500) - Italic

### Themes
- **Dark Theme (Default)**
  - Background: #1e1e1e
  - Foreground: #ffffff
  - Modern, eye-friendly appearance

- **Light Theme**
  - Background: #ffffff
  - Foreground: #000000
  - Clean, professional appearance

### Components
1. **Title Bar**: Shows "🌙 LunaAI Chat" with status indicator
2. **Status Label**: Displays "Loading...", "Model ready ✓", or "Generating response..."
3. **Chat Display**: Scrollable text area with color-coded messages
4. **Input Field**: Text entry with Enter key support
5. **Send Button**: Triggers message sending (disabled during generation)
6. **Action Buttons**:
   - Clear Chat: Clears conversation with confirmation
   - Toggle Theme: Switches between dark/light modes
   - Export Chat: Saves conversation to text file

### User Flow
1. Application launches → Shows loading message
2. Model loads asynchronously → Status updates to "Model ready ✓"
3. User types message and presses Enter or clicks Send
4. Message displays in green
5. Status changes to "Generating response..."
6. AI response displays in blue
7. Ready for next message

## Technical Implementation
- **Framework**: Tkinter with TTK widgets
- **Theme**: sv_ttk (Sun Valley theme)
- **Threading**: Async model loading and response generation
- **Layout**: Grid-based responsive design
- **Window Size**: 800x600 pixels (default)
- **Font**: Helvetica (11pt for messages, 18pt bold for title)
