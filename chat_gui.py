"""
LunaAI Chat GUI
Tkinter-based chat interface with sv_ttk theme for interacting with the trained model.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sv_ttk
import threading
import logging
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LunaAIChatGUI:
    """
    Chat GUI for LunaAI using Tkinter TTK with sv_ttk theme.
    """
    
    def __init__(self, model_path: str = "./luna_model"):
        """
        Initialize the chat GUI.
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.conversation_history = []
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("LunaAI Chat")
        self.root.geometry("800x600")
        
        # Apply sv_ttk theme
        sv_ttk.set_theme("dark")
        
        self._setup_ui()
        self._load_model_async()
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="🌙 LunaAI Chat",
            font=("Helvetica", 18, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Loading model...",
            font=("Helvetica", 10, "italic")
        )
        self.status_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.E)
        
        # Chat display area
        chat_frame = ttk.Frame(main_frame)
        chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            state=tk.DISABLED,
            background="#1e1e1e",
            foreground="#ffffff",
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags for styling
        self.chat_display.tag_config("user", foreground="#4CAF50", font=("Helvetica", 11, "bold"))
        self.chat_display.tag_config("assistant", foreground="#2196F3", font=("Helvetica", 11, "bold"))
        self.chat_display.tag_config("system", foreground="#FFA500", font=("Helvetica", 10, "italic"))
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        input_frame.columnconfigure(0, weight=1)
        
        # User input field
        self.user_input = ttk.Entry(input_frame, font=("Helvetica", 11))
        self.user_input.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.user_input.bind("<Return>", lambda e: self._send_message())
        
        # Send button
        self.send_button = ttk.Button(
            input_frame,
            text="Send",
            command=self._send_message,
            state=tk.DISABLED
        )
        self.send_button.grid(row=0, column=1)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=(10, 0))
        
        # Clear button
        clear_button = ttk.Button(
            button_frame,
            text="Clear Chat",
            command=self._clear_chat
        )
        clear_button.grid(row=0, column=0, padx=5)
        
        # Theme toggle button
        theme_button = ttk.Button(
            button_frame,
            text="Toggle Theme",
            command=self._toggle_theme
        )
        theme_button.grid(row=0, column=1, padx=5)
        
        # Export button
        export_button = ttk.Button(
            button_frame,
            text="Export Chat",
            command=self._export_chat
        )
        export_button.grid(row=0, column=2, padx=5)
    
    def _load_model_async(self):
        """Load the model asynchronously in a separate thread."""
        def load():
            try:
                self._append_system_message("Loading LunaAI model...")
                logger.info(f"Loading model from {self.model_path}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                )
                self.model.eval()
                
                self.model_loaded = True
                logger.info("Model loaded successfully")
                
                # Update UI
                self.root.after(0, self._on_model_loaded)
                
            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}"
                logger.error(error_msg)
                self.root.after(0, lambda: self._append_system_message(error_msg))
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _on_model_loaded(self):
        """Called when model is successfully loaded."""
        self.status_label.config(text="Model ready ✓")
        self.send_button.config(state=tk.NORMAL)
        self._append_system_message("Model loaded successfully! You can start chatting.")
        self.user_input.focus()
    
    def _append_message(self, role: str, message: str):
        """
        Append a message to the chat display.
        
        Args:
            role: Role of the message sender ('user', 'assistant', or 'system')
            message: Message content
        """
        self.chat_display.config(state=tk.NORMAL)
        
        if role == "system":
            self.chat_display.insert(tk.END, f"[SYSTEM] {message}\n\n", "system")
        else:
            role_tag = role
            role_display = "You" if role == "user" else "LunaAI"
            self.chat_display.insert(tk.END, f"{role_display}: ", role_tag)
            self.chat_display.insert(tk.END, f"{message}\n\n")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _append_system_message(self, message: str):
        """Append a system message."""
        self._append_message("system", message)
    
    def _send_message(self):
        """Handle sending a message."""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Model is still loading. Please wait.")
            return
        
        user_message = self.user_input.get().strip()
        if not user_message:
            return
        
        # Clear input
        self.user_input.delete(0, tk.END)
        
        # Display user message
        self._append_message("user", user_message)
        self.conversation_history.append(f"User: {user_message}")
        
        # Disable input while generating
        self.send_button.config(state=tk.DISABLED)
        self.user_input.config(state=tk.DISABLED)
        self.status_label.config(text="Generating response...")
        
        # Generate response in thread
        def generate():
            try:
                response = self._generate_response(user_message)
                self.root.after(0, lambda: self._on_response_generated(response))
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                self.root.after(0, lambda: self._on_response_generated(f"[Error: {str(e)}]"))
        
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
    
    def _generate_response(self, user_message: str) -> str:
        """
        Generate a response using the model.
        
        Args:
            user_message: User's input message
            
        Returns:
            Generated response
        """
        # Prepare input with conversation context
        context = "\n".join(self.conversation_history[-3:])  # Last 3 exchanges
        input_text = f"{context}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # Clean up response
        if "User:" in response:
            response = response.split("User:")[0].strip()
        
        return response if response else "I'm not sure how to respond to that."
    
    def _on_response_generated(self, response: str):
        """
        Called when response is generated.
        
        Args:
            response: Generated response
        """
        # Display assistant response
        self._append_message("assistant", response)
        self.conversation_history.append(f"Assistant: {response}")
        
        # Re-enable input
        self.send_button.config(state=tk.NORMAL)
        self.user_input.config(state=tk.NORMAL)
        self.status_label.config(text="Model ready ✓")
        self.user_input.focus()
    
    def _clear_chat(self):
        """Clear the chat history."""
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat?"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.conversation_history.clear()
            self._append_system_message("Chat cleared.")
    
    def _toggle_theme(self):
        """Toggle between light and dark themes."""
        current_theme = sv_ttk.get_theme()
        new_theme = "light" if current_theme == "dark" else "dark"
        sv_ttk.set_theme(new_theme)
        
        # Update chat display colors
        if new_theme == "dark":
            self.chat_display.config(background="#1e1e1e", foreground="#ffffff")
        else:
            self.chat_display.config(background="#ffffff", foreground="#000000")
    
    def _export_chat(self):
        """Export chat history to a file."""
        if not self.conversation_history:
            messagebox.showinfo("Export", "No chat history to export.")
            return
        
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("LunaAI Chat Export\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("\n".join(self.conversation_history))
                messagebox.showinfo("Export", f"Chat exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export chat: {str(e)}")
    
    def run(self):
        """Start the GUI event loop."""
        logger.info("Starting LunaAI Chat GUI")
        self.root.mainloop()


def main():
    """Main entry point for the chat GUI."""
    app = LunaAIChatGUI(model_path="./luna_model")
    app.run()


if __name__ == "__main__":
    main()
