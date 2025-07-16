# Invoice Q&A Assistant - Refactored Version

A modular RAG (Retrieval-Augmented Generation) application for querying invoice documents using hierarchical chunking and semantic search.

## 📁 Project Structure

```
RAG/
├── app.py                    # Main Streamlit application (refactored)
├── app_original.py          # Original monolithic version (backup)
├── config.py                # Configuration settings and prompts
├── database_manager.py      # ChromaDB operations and collection management
├── document_processor.py    # PDF processing and hierarchical chunking
├── ui_components.py         # Reusable Streamlit UI components
├── utils.py                 # Utility functions and helpers
├── callbacks.py             # Custom callback handlers for progress tracking
├── chroma_db/              # ChromaDB persistent storage
└── README.md               # This file
```

## 🏗️ Architecture Overview

### Core Modules

1. **`app.py`** - Main application entry point
   - `InvoiceQAApp` class orchestrates the entire application
   - Clean separation between UI rendering and business logic
   - Modular tab-based interface

2. **`config.py`** - Centralized configuration
   - `AppConfig`: Application settings and constants
   - `PromptTemplates`: LLM prompt templates
   - `FilterConfig`: Database query filters
   - `UIMessages`: Standardized UI text and labels

3. **`database_manager.py`** - Database operations
   - `DatabaseManager` class handles all ChromaDB operations
   - Collection management and querying
   - Hierarchical retrieval system
   - Collection statistics and analysis

4. **`document_processor.py`** - Document processing
   - `DocumentProcessor` class handles PDF processing
   - Hierarchical chunking (document + section levels)
   - Semantic chunking with progress tracking
   - Vector store creation with batch processing

5. **`ui_components.py`** - Reusable UI components
   - `CollectionSelector`: Collection picker interface
   - `QuickQuestions`: Pre-defined question buttons
   - `CollectionOverview`: Collections management tab
   - `QueryInterface`: Query processing and response display

6. **`utils.py`** - Utility functions
   - File handling utilities
   - Metadata formatting helpers
   - Progress tracking utilities
   - UI display helpers

7. **`callbacks.py`** - Custom callback handlers
   - `StreamlitProgressCallback`: Real-time progress tracking
   - `LoggingCallback`: Chain execution logging

## 🚀 Features

### Enhanced Functionality
- **Query without upload**: Use existing collections without uploading new files
- **Collection selector**: Easy switching between tenant/document type combinations
- **Progress tracking**: Detailed progress indicators for all operations
- **Hierarchical chunking**: Document and section-level chunks for better retrieval
- **Duplicate detection**: Prevents processing the same document twice
- **Collection management**: Overview and statistics for all collections

### Improved Code Quality
- **Modular architecture**: Clean separation of concerns
- **Type hints**: Better code documentation and IDE support
- **Error handling**: Robust error management throughout
- **Configuration management**: Centralized settings and prompts
- **Reusable components**: DRY principle implementation

## 🔧 Configuration

### Key Settings (config.py)

```python
# Database settings
CHROMA_PERSIST_DIR = "./chroma_db"

# Document processing
SEMANTIC_CHUNK_THRESHOLD = 75
RECURSIVE_CHUNK_SIZE = 512
BATCH_SIZE = 50

# LLM settings
DEFAULT_MODEL = "deepseek-r1:8b"
```

### Customizable Prompts

The invoice analysis prompt is centralized in `PromptTemplates.INVOICE_QA_PROMPT` and can be easily modified for different use cases.

## 🎯 Usage

### Basic Usage
```python
# Run the application
python app.py
```

### Using Individual Modules
```python
# Process documents
from document_processor import DocumentProcessor
processor = DocumentProcessor()
db, documents = processor.process_documents_with_progress(file, tenant_id, doc_type)

# Manage collections
from database_manager import DatabaseManager
db_manager = DatabaseManager()
collections = db_manager.get_available_collections()

# Use UI components
from ui_components import CollectionSelector
selector = CollectionSelector(collections)
tenant_id, doc_type = selector.render()
```

## 🔍 Key Improvements

### 1. **Maintainability**
- Separated concerns into focused modules
- Clear interfaces between components
- Easy to test individual components

### 2. **Extensibility**
- Easy to add new document types
- Pluggable UI components
- Configurable processing parameters

### 3. **Reusability**
- Components can be used independently
- Utility functions are module-agnostic
- Clean API design

### 4. **Performance**
- Efficient batch processing
- Progress tracking for better UX
- Optimized database operations

## 📊 Module Dependencies

```
app.py
├── config.py
├── database_manager.py
│   └── utils.py
├── document_processor.py
│   └── utils.py
├── ui_components.py
│   ├── utils.py
│   └── database_manager.py
└── callbacks.py
    └── config.py
```

## 🧪 Testing

Each module can be tested independently:

```python
# Test document processing
from document_processor import DocumentProcessor
processor = DocumentProcessor()

# Test database operations
from database_manager import DatabaseManager
db_manager = DatabaseManager()

# Test utilities
from utils import generate_file_hash, create_collection_name
```

## 🤝 Contributing

When adding new features:

1. **Follow the modular pattern**: Add functionality to the appropriate module
2. **Update configuration**: Add new settings to `config.py`
3. **Create reusable components**: Add UI components to `ui_components.py`
4. **Add utilities**: Common functions go in `utils.py`
5. **Document changes**: Update this README and add docstrings

## 📝 Migration Notes

### From Original Version
- All functionality is preserved
- Configuration is now centralized
- UI components are reusable
- Better error handling and progress tracking
- The original `app_original.py` is kept as backup

### Breaking Changes
- Import structure changed (now uses custom modules)
- Some internal function names changed
- Configuration moved to `config.py`

This refactored version maintains all original functionality while providing a much cleaner, more maintainable, and extensible codebase.