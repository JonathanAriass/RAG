"""
Configuration module for the RAG Invoice Q&A Assistant
Contains application settings, prompts, and model configurations
"""
from typing import Dict, Any
from langchain.prompts import PromptTemplate


class AppConfig:
    """Application configuration settings"""
    
    # Database settings
    CHROMA_PERSIST_DIR = "./chroma_db"
    
    # Document processing settings
    SEMANTIC_CHUNK_THRESHOLD = 75
    RECURSIVE_CHUNK_SIZE = 512
    RECURSIVE_CHUNK_OVERLAP = 50
    RECURSIVE_SEPARATORS = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    LARGE_SECTION_THRESHOLD = 800
    
    # Vector store settings
    BATCH_SIZE = 50
    SECTION_RETRIEVAL_K = 10
    DOCUMENT_RETRIEVAL_K = 2
    MAX_COMBINED_CHUNKS = 8
    
    # UI settings
    DOCUMENT_TYPES = ["invoice", "receipt", "bill", "statement"]
    DEFAULT_TENANT = "default_tenant"
    
    # Progress tracking weights
    PROGRESS_WEIGHTS = {
        'retrieval': 30,
        'llm_start': 20,
        'llm_processing': 40,
        'completion': 10
    }
    
    # LLM settings
    DEFAULT_MODEL = "deepseek-r1:8b"


class PromptTemplates:
    """Centralized prompt templates"""
    
    INVOICE_QA_PROMPT = """You are an expert invoice analyst. Extract and answer questions about invoice data based ONLY on the provided context.

INVOICE DATA EXPERTISE:
- Extract vendor/supplier information (name, address, contact details)
- Identify invoice numbers, dates, due dates
- Calculate totals, subtotals, taxes, discounts
- List line items with descriptions, quantities, unit prices
- Find payment terms, methods, and banking details
- Identify customer/buyer information

INSTRUCTIONS:
- Provide precise, structured answers using exact values from the invoice
- For monetary amounts, include currency symbols and exact figures
- For dates, use the exact format shown in the invoice
- If information is not in the context, state "This information is not found in the invoice"
- Quote specific sections when referencing data
- For calculations, show your work when relevant

Invoice Context:
{context}

Question: {question}

Answer:"""
    
    @classmethod
    def get_qa_prompt(cls) -> PromptTemplate:
        """Get the invoice Q&A prompt template"""
        return PromptTemplate.from_template(cls.INVOICE_QA_PROMPT)


class FilterConfig:
    """Configuration for database filters"""
    
    @staticmethod
    def get_section_filter(tenant_id: str, document_type: str) -> Dict[str, Any]:
        """Get filter for section-level chunks"""
        return {
            "$and": [
                {"tenant_id": {"$eq": tenant_id}},
                {"document_type": {"$eq": document_type}},
                {"chunk_type": {"$eq": "section"}}
            ]
        }
    
    @staticmethod
    def get_document_filter(tenant_id: str, document_type: str) -> Dict[str, Any]:
        """Get filter for document-level chunks"""
        return {
            "$and": [
                {"tenant_id": {"$eq": tenant_id}},
                {"document_type": {"$eq": document_type}},
                {"chunk_type": {"$eq": "document"}}
            ]
        }


class UIMessages:
    """Centralized UI messages and labels"""
    
    # Status messages
    LOADING_PDF = "📄 Loading PDF document..."
    EXTRACTING_TEXT = "📖 Extracting text from PDF..."
    ANALYZING_STRUCTURE = "🏗️ Analyzing document structure..."
    CREATING_CHUNKS = "🧠 Creating semantic chunks..."
    CREATING_VECTORSTORE = "🗄️ Creating vector database..."
    PROCESSING_COMPLETE = "✅ Processing complete!"
    
    # Error messages
    ERROR_NO_COLLECTION = "❌ No collection available. Please upload a document first."
    ERROR_PROCESSING = "❌ Error during processing"
    
    # Info messages
    COLLECTION_EXISTS = "📁 **Found existing collection:**"
    QUERY_WITHOUT_UPLOAD = "✨ **You can query this collection directly without uploading a file!**"
    NO_COLLECTION_FOUND = "📄 **No existing collection found.** Please upload a document to create one."
    ADDING_NEW_DOCUMENT = "📄 Adding new document to existing collection..."
    PROCESSING_NEW_DOCUMENT = "📄 Processing new document for new collection..."
    
    # Query processing messages
    RETRIEVING_DOCS = "🔍 Retrieving relevant documents..."
    PREPARING_CONTEXT = "📋 Preparing context for analysis..."
    AI_ANALYZING = "🤖 Analyzing with AI model..."
    RESPONSE_READY = "✅ Response ready!"
    
    # Quick question labels
    QUICK_QUESTIONS = {
        "date_number": "📅 Invoice Date & Number",
        "total_amount": "💰 Total Amount",
        "vendor_info": "🏢 Vendor Information",
        "line_items": "📋 Line Items",
        "payment_terms": "💳 Payment Terms",
        "tax_breakdown": "🧮 Tax Breakdown"
    }
    
    # Quick question queries
    QUICK_QUERIES = {
        "date_number": "What is the invoice number and invoice date?",
        "total_amount": "What is the total amount of this invoice?",
        "vendor_info": "Who is the vendor/supplier and their contact details?",
        "line_items": "List all line items with quantities and prices",
        "payment_terms": "What are the payment terms and due date?",
        "tax_breakdown": "Show the tax breakdown and subtotals"
    }


class MetadataFields:
    """Standardized metadata field names"""
    
    # Core fields
    TENANT_ID = "tenant_id"
    DOCUMENT_TYPE = "document_type"
    UPLOAD_TIMESTAMP = "upload_timestamp"
    SOURCE_FILENAME = "source_filename"
    FILE_HASH = "file_hash"
    NAMESPACE = "namespace"
    
    # Chunk fields
    CHUNK_TYPE = "chunk_type"
    CHUNK_LEVEL = "chunk_level"
    DOCUMENT_ID = "document_id"
    PARENT_ID = "parent_id"
    PAGE_NUMBER = "page_number"
    TOTAL_CHARS = "total_chars"
    
    # Section fields
    SECTION_ID = "section_id"
    SECTION_INDEX = "section_index"
    SUBSECTION_INDEX = "subsection_index"
    
    # Chunk types
    CHUNK_TYPE_DOCUMENT = "document"
    CHUNK_TYPE_SECTION = "section"


def get_app_title() -> str:
    """Get the application title"""
    return "🧾 Invoice Q&A Assistant"


def get_app_description() -> str:
    """Get the application description"""
    return "Upload an invoice PDF and ask questions about vendor details, amounts, dates, line items, and more."


def get_tab_names() -> tuple[str, str]:
    """Get tab names for the application"""
    return "📄 Document Q&A", "🗄️ Collections Overview"