"""
LunaAI GUI - Tkinter interface for interacting with HuggingFace datasets
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import logging

try:
    from huggingface_datasets import HuggingFaceDatasetLoader
except ImportError as e:
    raise ImportError("Failed to import huggingface_datasets. Ensure the module is in the same directory.") from e

try:
    from luna_model import SimpleLunaAI, LunaAI
except ImportError as e:
    logger.warning("Failed to import luna_model. Using basic search only.")
    SimpleLunaAI = None
    LunaAI = None

try:
    import sv_ttk
except ImportError as e:
    raise ImportError("sv_ttk is required. Install it with: pip install sv-ttk") from e

try:
    from huggingface_hub import login, whoami
    from huggingface_hub import get_token
    HF_HUB_AVAILABLE = True
except ImportError as e:
    logger.warning("huggingface_hub functions not fully available")
    HF_HUB_AVAILABLE = False
    login = None
    whoami = None
    get_token = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LunaAIGUI:
    """Main GUI application for LunaAI"""
    
    # Configuration constants
    MAX_SEARCH_ITEMS = 1000  # Maximum items to search through
    DEFAULT_RESULTS = 5  # Default number of results to show
    
    def __init__(self, root):
        self.root = root
        self.root.title("LunaAI - HuggingFace Dataset Explorer")
        self.root.geometry("900x700")
        
        # Apply sv_ttk theme with error handling
        try:
            sv_ttk.set_theme("dark")
        except Exception as e:
            logger.warning(f"Failed to apply sv_ttk theme: {e}")
            messagebox.showwarning("Theme Error", "Failed to apply theme. Using default.")
        
        # Initialize dataset loader
        self.loader = HuggingFaceDatasetLoader()
        self.loaded_datasets = {}
        self.current_dataset_key = None
        self.is_logged_in = False
        
        # Initialize Luna AI model
        if SimpleLunaAI:
            self.luna = SimpleLunaAI()
            logger.info("Luna AI initialized in simple mode")
        else:
            self.luna = None
            logger.warning("Luna AI not available")
        
        # Check initial login status
        self.check_login_status()
        
        # Setup GUI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title frame
        self.setup_title_frame(main_frame)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: Dataset Explorer
        dataset_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(dataset_tab, text="Dataset Explorer")
        self.setup_dataset_tab(dataset_tab)
        
        # Tab 2: Chat with Luna
        chat_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(chat_tab, text="Chat with Luna")
        self.setup_chat_tab(chat_tab)
        
        # Tab 3: Train Luna
        train_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(train_tab, text="Train Luna")
        self.setup_train_tab(train_tab)
        
        # Status bar (bottom of main frame)
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN,
                                   anchor=tk.W)
        self.status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def setup_dataset_tab(self, parent):
        """Setup the dataset explorer tab"""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        
        # Title and theme toggle
        title_frame = ttk.Frame(parent)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        title_frame.columnconfigure(0, weight=1)
        
        title_label = ttk.Label(title_frame, text="LunaAI Dataset Explorer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # HuggingFace login status
        self.login_status_label = ttk.Label(title_frame, text="HF: Not logged in", 
                                           foreground='orange')
        self.login_status_label.grid(row=0, column=1, padx=(10, 5))
        
        # Login button
        self.login_button = ttk.Button(title_frame, text="Login to HuggingFace",
                                      command=self.show_login_dialog)
        self.login_button.grid(row=0, column=2, padx=(0, 5))
        
        # Theme toggle button
        self.theme_button = ttk.Button(title_frame, text="Toggle Theme",
                                      command=self.toggle_theme)
        self.theme_button.grid(row=0, column=3, sticky=tk.E)
        self.current_theme = "dark"
        
        # Dataset selection frame
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset Selection", padding="10")
        dataset_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        dataset_frame.columnconfigure(0, weight=1)
        
        # Dataset dropdown
        dataset_label = ttk.Label(dataset_frame, text="Select Dataset:")
        dataset_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.dataset_var = tk.StringVar()
        available_datasets = self.loader.list_available_datasets()
        self.dataset_dropdown = ttk.Combobox(dataset_frame, textvariable=self.dataset_var,
                                            values=available_datasets, state='readonly',
                                            width=40)
        self.dataset_dropdown.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        if available_datasets:
            self.dataset_dropdown.set(available_datasets[0])
        
        # Dataset info display
        self.dataset_info_label = ttk.Label(dataset_frame, text="", foreground='blue')
        self.dataset_info_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        # Update info when selection changes
        self.dataset_dropdown.bind('<<ComboboxSelected>>', self.update_dataset_info)
        self.update_dataset_info()
        
        # Load buttons frame
        buttons_frame = ttk.Frame(dataset_frame)
        buttons_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        self.load_button = ttk.Button(buttons_frame, text="Load Selected Dataset",
                                     command=self.load_selected_dataset)
        self.load_button.grid(row=0, column=0, padx=(0, 5))
        
        self.load_all_button = ttk.Button(buttons_frame, text="Load All Datasets",
                                         command=self.load_all_datasets)
        self.load_all_button.grid(row=0, column=1)
        
        # Loaded datasets display
        loaded_label = ttk.Label(dataset_frame, text="Loaded datasets:")
        loaded_label.grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        
        self.loaded_text = tk.Text(dataset_frame, height=3, width=40, state='disabled')
        self.loaded_text.grid(row=5, column=0, sticky=(tk.W, tk.E))
        
        # Query frame
        query_frame = ttk.LabelFrame(main_frame, text="Query Dataset", padding="10")
        query_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        query_frame.columnconfigure(0, weight=1)
        
        query_label = ttk.Label(query_frame, text="Enter your query:")
        query_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.query_entry = ttk.Entry(query_frame, width=50)
        self.query_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.query_entry.bind('<Return>', lambda e: self.query_dataset())
        
        # Query controls
        query_controls = ttk.Frame(query_frame)
        query_controls.grid(row=2, column=0, sticky=tk.W)
        
        self.query_button = ttk.Button(query_controls, text="Query Dataset",
                                      command=self.query_dataset)
        self.query_button.grid(row=0, column=0, padx=(0, 5))
        
        self.clear_button = ttk.Button(query_controls, text="Clear Output",
                                      command=self.clear_output)
        self.clear_button.grid(row=0, column=1)
        
        # Number of results
        results_label = ttk.Label(query_controls, text="Results to show:")
        results_label.grid(row=0, column=2, padx=(10, 5))
        
        self.results_var = tk.StringVar(value=str(self.DEFAULT_RESULTS))
        results_spinbox = ttk.Spinbox(query_controls, from_=1, to=50, width=5,
                                     textvariable=self.results_var)
        results_spinbox.grid(row=0, column=3)
        
        # Output frame
        output_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        # Scrolled text for output
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD,
                                                     width=80, height=15)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN,
                                   anchor=tk.W)
        self.status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def update_dataset_info(self, event=None):
        """Update dataset information display"""
        dataset_key = self.dataset_var.get()
        if dataset_key:
            try:
                info = self.loader.get_dataset_info(dataset_key)
                info_text = f"{info['name']} - {info['description']}"
                self.dataset_info_label.config(text=info_text)
            except Exception as e:
                self.dataset_info_label.config(text="")
    
    def update_loaded_datasets_display(self):
        """Update the display of loaded datasets"""
        self.loaded_text.config(state='normal')
        self.loaded_text.delete(1.0, tk.END)
        if self.loaded_datasets:
            loaded_names = ', '.join(self.loaded_datasets.keys())
            self.loaded_text.insert(1.0, loaded_names)
        else:
            self.loaded_text.insert(1.0, "No datasets loaded")
        self.loaded_text.config(state='disabled')
    
    def set_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def append_output(self, text):
        """Append text to output area"""
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_output(self):
        """Clear the output text area"""
        self.output_text.delete(1.0, tk.END)
    
    def disable_buttons(self):
        """Disable all action buttons"""
        self.load_button.config(state='disabled')
        self.load_all_button.config(state='disabled')
        self.query_button.config(state='disabled')
    
    def enable_buttons(self):
        """Enable all action buttons"""
        self.load_button.config(state='normal')
        self.load_all_button.config(state='normal')
        self.query_button.config(state='normal')
    
    def load_selected_dataset(self):
        """Load the selected dataset in a background thread"""
        dataset_key = self.dataset_var.get()
        if not dataset_key:
            messagebox.showwarning("No Selection", "Please select a dataset to load")
            return
        
        def load_thread():
            try:
                self.disable_buttons()
                self.set_status(f"Loading {dataset_key}...")
                self.append_output(f"\n{'='*60}")
                self.append_output(f"Loading dataset: {dataset_key}")
                self.append_output(f"{'='*60}")
                
                # Whitelist of allowed dataset loading methods
                allowed_loaders = {
                    'maptrace': self.loader.load_maptrace,
                    'diffusiondb': self.loader.load_diffusiondb,
                    'websight': self.loader.load_websight,
                    'community_dataset': self.loader.load_community_dataset,
                    'finevision': self.loader.load_finevision,
                    'cads': self.loader.load_cads,
                    'synth': self.loader.load_synth,
                    'wikipedia': self.loader.load_wikipedia,
                    'deepmath': self.loader.load_deepmath,
                    'acemath': self.loader.load_acemath,
                    'smoltalk': self.loader.load_smoltalk,
                    'superior_reasoning': self.loader.load_superior_reasoning,
                    'audioskills': self.loader.load_audioskills,
                    'mobile_actions': self.loader.load_mobile_actions
                }
                
                if dataset_key in allowed_loaders:
                    loader_method = allowed_loaders[dataset_key]
                    dataset = loader_method(streaming=True)
                    self.loaded_datasets[dataset_key] = dataset
                    self.current_dataset_key = dataset_key
                    
                    self.append_output(f"✓ Successfully loaded {dataset_key}")
                    self.append_output(f"Dataset type: {type(dataset).__name__}")
                    self.update_loaded_datasets_display()
                    self.set_status(f"Ready - {dataset_key} loaded")
                else:
                    self.append_output(f"✗ Error: Unknown dataset key {dataset_key}")
                    self.set_status("Error loading dataset")
                    
            except Exception as e:
                error_msg = f"✗ Error loading {dataset_key}: {str(e)}"
                self.append_output(error_msg)
                self.set_status("Error loading dataset")
                logger.error(error_msg)
            finally:
                self.enable_buttons()
        
        # Run in background thread
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def load_all_datasets(self):
        """Load all datasets in a background thread"""
        
        def load_thread():
            try:
                self.disable_buttons()
                self.set_status("Loading all datasets...")
                self.append_output(f"\n{'='*60}")
                self.append_output("Loading all datasets...")
                self.append_output(f"{'='*60}")
                
                # Load all datasets
                all_datasets = self.loader.load_all_datasets(streaming=True)
                self.loaded_datasets.update(self.loader.get_loaded_datasets())
                
                if self.loaded_datasets:
                    if not self.current_dataset_key or self.current_dataset_key not in self.loaded_datasets:
                        self.current_dataset_key = list(self.loaded_datasets.keys())[0]
                
                self.append_output(f"\n✓ Successfully loaded {len(self.loaded_datasets)} datasets:")
                for key in self.loaded_datasets.keys():
                    self.append_output(f"  • {key}")
                
                self.update_loaded_datasets_display()
                self.set_status(f"Ready - {len(self.loaded_datasets)} datasets loaded")
                
            except Exception as e:
                error_msg = f"✗ Error loading datasets: {str(e)}"
                self.append_output(error_msg)
                self.set_status("Error loading datasets")
                logger.error(error_msg)
            finally:
                self.enable_buttons()
        
        # Run in background thread
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def query_dataset(self):
        """Query the loaded dataset using Luna AI"""
        if not self.loaded_datasets:
            messagebox.showwarning("No Dataset", "Please load a dataset first")
            return
        
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a query")
            return
        
        def query_thread():
            try:
                self.disable_buttons()
                self.set_status("Luna is processing your query...")
                
                num_results = int(self.results_var.get())
                
                self.append_output(f"\n{'='*60}")
                self.append_output(f"Query: {query}")
                self.append_output(f"Using Luna AI to analyze: {self.current_dataset_key or 'all loaded datasets'}")
                self.append_output(f"{'='*60}\n")
                
                # Get the current dataset to query
                if self.current_dataset_key and self.current_dataset_key in self.loaded_datasets:
                    dataset = self.loaded_datasets[self.current_dataset_key]
                    self.query_with_luna(dataset, query, num_results, self.current_dataset_key)
                else:
                    # Query all datasets
                    for key, dataset in self.loaded_datasets.items():
                        self.append_output(f"\n Luna analyzing {key}:")
                        self.append_output("-" * 40)
                        self.query_with_luna(dataset, query, num_results, key)
                
                self.set_status("Query complete")
                
            except Exception as e:
                error_msg = f"✗ Error querying dataset: {str(e)}"
                self.append_output(error_msg)
                self.set_status("Error during query")
                logger.error(error_msg)
            finally:
                self.enable_buttons()
        
        # Run in background thread
        thread = threading.Thread(target=query_thread, daemon=True)
        thread.start()
    
    def query_with_luna(self, dataset, query, num_results, dataset_name):
        """Use Luna AI to intelligently query the dataset"""
        try:
            # Collect samples from the dataset
            dataset_samples = []
            max_samples = min(num_results * 2, 20)  # Get more samples for context
            
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                dataset_samples.append(item)
            
            if not dataset_samples:
                self.append_output(f"No samples available from {dataset_name}")
                return
            
            # Use Luna AI if available
            if self.luna and self.luna.is_loaded:
                self.append_output("\nLuna AI is analyzing the dataset...")
                response = self.luna.query_dataset(query, dataset_samples, dataset_name, max_samples)
                self.append_output(response)
            else:
                # Fallback to basic search
                self.search_dataset(dataset, query, num_results, dataset_name)
                
        except Exception as e:
            error_msg = f"Error with Luna: {str(e)}"
            self.append_output(error_msg)
            logger.error(error_msg)
    
    def search_dataset(self, dataset, query, num_results, dataset_name):
        """Search through dataset and display results"""
        try:
            query_lower = query.lower()
            results_found = 0
            items_checked = 0
            
            for item in dataset:
                items_checked += 1
                if items_checked > self.MAX_SEARCH_ITEMS:
                    self.append_output(f"\n(Searched {self.MAX_SEARCH_ITEMS} items, stopping search)")
                    break
                
                # Convert item to string and search
                item_str = str(item).lower()
                if query_lower in item_str:
                    results_found += 1
                    self.append_output(f"\nResult {results_found}:")
                    
                    # Display the item in a readable format
                    if isinstance(item, dict):
                        for key, value in item.items():
                            # Truncate long values
                            value_str = str(value)
                            if len(value_str) > 200:
                                value_str = value_str[:200] + "..."
                            self.append_output(f"  {key}: {value_str}")
                    else:
                        item_str = str(item)
                        if len(item_str) > 500:
                            item_str = item_str[:500] + "..."
                        self.append_output(f"  {item_str}")
                    
                    if results_found >= num_results:
                        break
            
            if results_found == 0:
                self.append_output(f"No results found for '{query}' in {dataset_name}")
                self.append_output(f"(Searched {items_checked} items)")
            else:
                self.append_output(f"\nFound {results_found} results in {dataset_name}")
                
        except Exception as e:
            error_msg = f"Error searching {dataset_name}: {str(e)}"
            self.append_output(error_msg)
            logger.error(error_msg)
    
    def check_login_status(self):
        """Check if user is logged into HuggingFace"""
        if not HF_HUB_AVAILABLE:
            self.is_logged_in = False
            return
        
        try:
            # Try to get the token
            token = get_token()
            if token:
                # Try to verify with whoami
                try:
                    user_info = whoami(token)
                    username = user_info.get('name', 'Unknown')
                    self.is_logged_in = True
                    logger.info(f"Logged in to HuggingFace as: {username}")
                    if hasattr(self, 'login_status_label'):
                        self.login_status_label.config(text=f"HF: {username}", foreground='green')
                        self.login_button.config(text="Logout")
                except Exception as e:
                    self.is_logged_in = False
                    logger.warning(f"Token exists but verification failed: {e}")
            else:
                self.is_logged_in = False
                if hasattr(self, 'login_status_label'):
                    self.login_status_label.config(text="HF: Not logged in", foreground='orange')
        except Exception as e:
            self.is_logged_in = False
            logger.error(f"Error checking login status: {e}")
    
    def show_login_dialog(self):
        """Show login dialog or logout"""
        if not HF_HUB_AVAILABLE:
            messagebox.showerror("Error", "HuggingFace Hub not available. Install with: pip install huggingface_hub")
            return
        
        if self.is_logged_in:
            # Logout
            response = messagebox.askyesno("Logout", "Do you want to logout from HuggingFace?")
            if response:
                try:
                    # Logout by passing token=False
                    from huggingface_hub import logout
                    logout()
                    self.is_logged_in = False
                    self.login_status_label.config(text="HF: Not logged in", foreground='orange')
                    self.login_button.config(text="Login to HuggingFace")
                    messagebox.showinfo("Success", "Logged out successfully")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to logout: {e}")
        else:
            # Show login dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Login to HuggingFace")
            dialog.geometry("400x200")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
            
            frame = ttk.Frame(dialog, padding="20")
            frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(frame, text="Enter your HuggingFace access token:",
                     font=('Arial', 10)).pack(pady=(0, 10))
            
            token_entry = ttk.Entry(frame, width=50, show="*")
            token_entry.pack(pady=(0, 10))
            token_entry.focus()
            
            ttk.Label(frame, text="Get your token from: https://huggingface.co/settings/tokens",
                     foreground='blue', font=('Arial', 8)).pack(pady=(0, 10))
            
            result = {'success': False}
            
            def do_login():
                token = token_entry.get().strip()
                if not token:
                    messagebox.showwarning("Empty Token", "Please enter a token")
                    return
                
                try:
                    # Try to login
                    login(token=token)
                    
                    # Verify login
                    user_info = whoami(token)
                    username = user_info.get('name', 'Unknown')
                    
                    self.is_logged_in = True
                    self.login_status_label.config(text=f"HF: {username}", foreground='green')
                    self.login_button.config(text="Logout")
                    
                    result['success'] = True
                    messagebox.showinfo("Success", f"Logged in as: {username}")
                    dialog.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Login Failed", f"Failed to login: {str(e)}")
            
            button_frame = ttk.Frame(frame)
            button_frame.pack(pady=(10, 0))
            
            ttk.Button(button_frame, text="Login", command=do_login).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            
            # Bind Enter key to login
            token_entry.bind('<Return>', lambda e: do_login())
            
            dialog.wait_window()
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        try:
            if self.current_theme == "dark":
                sv_ttk.set_theme("light")
                self.current_theme = "light"
            else:
                sv_ttk.set_theme("dark")
                self.current_theme = "dark"
        except Exception as e:
            logger.error(f"Failed to toggle theme: {e}")
            messagebox.showerror("Theme Error", f"Failed to toggle theme: {str(e)}")


def main():
    """Main entry point for the GUI application"""
    root = tk.Tk()
    app = LunaAIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
