"""
LunaAI GUI - Tkinter interface for interacting with HuggingFace datasets
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import logging
from huggingface_datasets import HuggingFaceDatasetLoader
import sv_ttk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LunaAIGUI:
    """Main GUI application for LunaAI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("LunaAI - HuggingFace Dataset Explorer")
        self.root.geometry("900x700")
        
        # Apply sv_ttk theme
        sv_ttk.set_theme("dark")
        
        # Initialize dataset loader
        self.loader = HuggingFaceDatasetLoader()
        self.loaded_datasets = {}
        self.current_dataset_key = None
        
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
        main_frame.rowconfigure(3, weight=1)
        
        # Title and theme toggle
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        title_frame.columnconfigure(0, weight=1)
        
        title_label = ttk.Label(title_frame, text="LunaAI Dataset Explorer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Theme toggle button
        self.theme_button = ttk.Button(title_frame, text="Toggle Theme",
                                      command=self.toggle_theme)
        self.theme_button.grid(row=0, column=1, sticky=tk.E)
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
        
        self.loaded_text = tk.Text(dataset_frame, height=3, width=40, state='disabled',
                                  background='#f0f0f0')
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
        
        self.results_var = tk.StringVar(value="5")
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
                
                # Load the dataset based on key
                method_name = f"load_{dataset_key}"
                if hasattr(self.loader, method_name):
                    method = getattr(self.loader, method_name)
                    dataset = method(streaming=True)
                    self.loaded_datasets[dataset_key] = dataset
                    self.current_dataset_key = dataset_key
                    
                    self.append_output(f"✓ Successfully loaded {dataset_key}")
                    self.append_output(f"Dataset type: {type(dataset).__name__}")
                    self.update_loaded_datasets_display()
                    self.set_status(f"Ready - {dataset_key} loaded")
                else:
                    self.append_output(f"✗ Error: No loader method found for {dataset_key}")
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
        """Query the loaded dataset"""
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
                self.set_status("Querying dataset...")
                
                num_results = int(self.results_var.get())
                
                self.append_output(f"\n{'='*60}")
                self.append_output(f"Query: {query}")
                self.append_output(f"Searching in: {self.current_dataset_key or 'all loaded datasets'}")
                self.append_output(f"{'='*60}\n")
                
                # Get the current dataset to query
                if self.current_dataset_key and self.current_dataset_key in self.loaded_datasets:
                    dataset = self.loaded_datasets[self.current_dataset_key]
                    self.search_dataset(dataset, query, num_results, self.current_dataset_key)
                else:
                    # Query all datasets
                    for key, dataset in self.loaded_datasets.items():
                        self.append_output(f"\nSearching in {key}:")
                        self.append_output("-" * 40)
                        self.search_dataset(dataset, query, num_results, key)
                
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
    
    def search_dataset(self, dataset, query, num_results, dataset_name):
        """Search through dataset and display results"""
        try:
            query_lower = query.lower()
            results_found = 0
            items_checked = 0
            max_items_to_check = 1000  # Limit search to first 1000 items
            
            for item in dataset:
                items_checked += 1
                if items_checked > max_items_to_check:
                    self.append_output(f"\n(Searched {max_items_to_check} items, stopping search)")
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
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        if self.current_theme == "dark":
            sv_ttk.set_theme("light")
            self.current_theme = "light"
        else:
            sv_ttk.set_theme("dark")
            self.current_theme = "dark"


def main():
    """Main entry point for the GUI application"""
    root = tk.Tk()
    app = LunaAIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
