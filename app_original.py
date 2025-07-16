import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler
import datetime
import uuid
import time

# Custom Progress Callback Handler for LangChain
class StreamlitProgressCallback(BaseCallbackHandler):
    def __init__(self, progress_bar, status_text, stage_weights=None):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.current_progress = 0
        self.stage_weights = stage_weights or {
            'retrieval': 30,
            'llm_start': 20,
            'llm_processing': 40,
            'completion': 10
        }
    
    def on_retriever_start(self, serialized, query, **kwargs):
        self.status_text.text("🔍 Searching through invoice documents...")
        self.current_progress += self.stage_weights['retrieval']
        self.progress_bar.progress(min(self.current_progress, 95))
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.status_text.text("🤖 Starting AI analysis...")
        self.current_progress += self.stage_weights['llm_start']
        self.progress_bar.progress(min(self.current_progress, 95))
    
    def on_llm_new_token(self, token: str, **kwargs):
        if hasattr(self, '_token_count'):
            self._token_count += 1
        else:
            self._token_count = 1
        
        # Update progress based on token generation
        if self._token_count % 5 == 0:  # Update every 5 tokens to avoid too frequent updates
            token_progress = min(self._token_count / 200, 1.0) * self.stage_weights['llm_processing']
            self.progress_bar.progress(min(self.current_progress + token_progress, 95))
    
    def on_llm_end(self, response, **kwargs):
        self.status_text.text("✅ Analysis complete!")
        self.current_progress += self.stage_weights['completion']
        self.progress_bar.progress(100)

# Enhanced document processing with progress tracking
def process_documents_with_progress(uploaded_file, tenant_id, document_type):
    """Process uploaded PDF with detailed progress tracking"""
    
    # Initialize progress tracking
    progress_container = st.container()
    with progress_container:
        main_progress = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
    
    try:
        # Stage 1: File Processing (20%)
        status_text.text("📄 Loading PDF document...")
        main_progress.progress(10)
        
        # Create unique temporary filename to avoid collisions
        import hashlib
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
        temp_filename = f"temp_{file_hash}_{uploaded_file.name.replace(' ', '_')}"
        
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = PDFPlumberLoader(temp_filename)
        status_text.text("📖 Extracting text from PDF...")
        main_progress.progress(20)
        docs = loader.load()
        
        # Stage 2: Document Structure Analysis (40%)
        status_text.text("🏗️ Analyzing document structure...")
        main_progress.progress(30)
        
        hierarchical_documents = []
        total_docs = len(docs)
        
        # Process each document with sub-progress
        for doc_idx, doc in enumerate(docs):
            detail_text.text(f"Processing page {doc_idx + 1} of {total_docs}")
            
            doc_id = str(uuid.uuid4())
            
            # Create document-level chunk
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            doc_chunk = Document(
                page_content=doc.page_content,
                metadata={
                    "tenant_id": tenant_id,
                    "document_type": document_type,
                    "upload_timestamp": datetime.datetime.now().isoformat(),
                    "source_filename": uploaded_file.name,
                    "file_hash": file_hash,
                    "namespace": f"{tenant_id}_{document_type}",
                    "chunk_type": "document",
                    "chunk_level": 1,
                    "document_id": doc_id,
                    "parent_id": None,
                    "page_number": doc.metadata.get("page", doc_idx + 1),
                    "total_chars": len(doc.page_content)
                }
            )
            hierarchical_documents.append(doc_chunk)
            
            # Update progress for document processing
            doc_progress = 30 + (doc_idx + 1) / total_docs * 20
            main_progress.progress(int(doc_progress))
        
        # Stage 3: Semantic Chunking (60%)
        status_text.text("🧠 Creating semantic chunks...")
        main_progress.progress(50)
        
        # Initialize embeddings with progress
        detail_text.text("Loading embedding model...")
        embedder = HuggingFaceEmbeddings()
        
        # Process semantic chunking
        for doc_idx, doc in enumerate(docs):
            detail_text.text(f"Semantic analysis of page {doc_idx + 1}")
            
            doc_id = hierarchical_documents[doc_idx].metadata["document_id"]
            
            # Use semantic chunker
            semantic_splitter = SemanticChunker(
                HuggingFaceEmbeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=75
            )
            section_chunks = semantic_splitter.split_documents([doc])
            
            # Process sections with sub-chunking if needed
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            for section_idx, section in enumerate(section_chunks):
                if len(section.page_content) > 800:
                    # Further split large sections
                    subsections = recursive_splitter.split_documents([section])
                    for subsection_idx, subsection in enumerate(subsections):
                        section_chunk = Document(
                            page_content=subsection.page_content,
                            metadata={
                                "tenant_id": tenant_id,
                                "document_type": document_type,
                                "upload_timestamp": (
                                    datetime.datetime.now().isoformat()
                                ),
                                "source_filename": uploaded_file.name,
                                "file_hash": file_hash,
                                "namespace": f"{tenant_id}_{document_type}",
                                "chunk_type": "section",
                                "chunk_level": 2,
                                "document_id": doc_id,
                                "parent_id": doc_id,
                                "section_id": (
                                    f"{doc_id}_s{section_idx}_sub{subsection_idx}"
                                ),
                                "page_number": doc.metadata.get("page", doc_idx + 1),
                                "section_index": section_idx,
                                "subsection_index": subsection_idx,
                                "total_chars": len(subsection.page_content)
                            }
                        )
                        hierarchical_documents.append(section_chunk)
                else:
                    section_chunk = Document(
                        page_content=section.page_content,
                        metadata={
                            "tenant_id": tenant_id,
                            "document_type": document_type,
                            "upload_timestamp": (
                                datetime.datetime.now().isoformat()
                            ),
                            "source_filename": uploaded_file.name,
                            "file_hash": file_hash,
                            "namespace": f"{tenant_id}_{document_type}",
                            "chunk_type": "section",
                            "chunk_level": 2,
                            "document_id": doc_id,
                            "parent_id": doc_id,
                            "section_id": f"{doc_id}_s{section_idx}",
                            "page_number": doc.metadata.get("page", doc_idx + 1),
                            "section_index": section_idx,
                            "total_chars": len(section.page_content)
                        }
                    )
                    hierarchical_documents.append(section_chunk)
            
            # Update progress for semantic processing
            semantic_progress = 50 + (doc_idx + 1) / total_docs * 20
            main_progress.progress(int(semantic_progress))
        
        # Stage 4: Vector Store Creation (80%)
        status_text.text("🗄️ Creating vector database...")
        main_progress.progress(70)
        
        collection_name = f"tenant_{tenant_id}_{document_type}"
        detail_text.text(f"Building collection: {collection_name}")
        
        # Create vector store with batch processing for better progress tracking
        batch_size = 50
        total_batches = len(hierarchical_documents) // batch_size + (1 if len(hierarchical_documents) % batch_size else 0)
        
        db = None
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(hierarchical_documents))
            batch_docs = hierarchical_documents[start_idx:end_idx]
            
            detail_text.text(f"Processing batch {batch_idx + 1} of {total_batches}")
            
            if db is None:
                db = Chroma.from_documents(
                    batch_docs,
                    embedder,
                    persist_directory="./chroma_db",
                    collection_name=collection_name
                )
            else:
                db.add_documents(batch_docs)
            
            # Update progress for vector store creation
            batch_progress = 70 + (batch_idx + 1) / total_batches * 25
            main_progress.progress(int(batch_progress))
        
        # Stage 5: Finalization (100%)
        status_text.text("✅ Processing complete!")
        detail_text.text(f"Created {len(hierarchical_documents)} chunks in {total_batches} batches")
        main_progress.progress(100)
        
        # Brief pause to show completion
        time.sleep(1)
        
        # Cleanup temporary file
        import os
        try:
            os.remove(temp_filename)
        except OSError:
            pass  # Ignore cleanup errors
        
        # Clear progress indicators
        progress_container.empty()
        
        return db, hierarchical_documents
        
    except Exception as e:
        status_text.text(f"❌ Error during processing: {str(e)}")
        main_progress.progress(0)
        raise e

# Streamlit UI
st.title("🧾 Invoice Q&A Assistant")
st.markdown("Upload an invoice PDF and ask questions about vendor details, amounts, dates, line items, and more.")

# Create tabs
tab1, tab2 = st.tabs(["📄 Document Q&A", "🗄️ Collections Overview"])

with tab1:
    st.markdown("### Document Q&A Interface")
    
    # Get available collections for easier selection
    available_collections = []
    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collections = chroma_client.list_collections()
        for collection in collections:
            if collection.name.startswith("tenant_"):
                coll = chroma_client.get_collection(collection.name)
                if coll.count() > 0:
                    available_collections.append(collection.name)
    except Exception:
        pass
    
    # Show available collections if any exist
    if available_collections:
        st.info(f"📁 **Available collections:** {len(available_collections)} collections found")
        
        use_existing = st.checkbox("🔍 **Select from existing collections**", value=True)
        
        if use_existing:
            selected_collection = st.selectbox("Choose collection:", available_collections)
            if selected_collection:
                # Parse collection name to extract tenant_id and document_type
                parts = selected_collection.split("_")
                if len(parts) >= 3:
                    tenant_id = parts[1]
                    document_type = "_".join(parts[2:])
                else:
                    tenant_id = "default_tenant"
                    document_type = "invoice"
            else:
                tenant_id = "default_tenant"
                document_type = "invoice"
        else:
            # Manual input
            tenant_id = st.text_input("Tenant ID", value="default_tenant", help="Enter your tenant/namespace identifier")
            document_type = st.selectbox("Document Type", ["invoice", "receipt", "bill", "statement"], help="Select document category")
    else:
        # No existing collections, use manual input
        tenant_id = st.text_input("Tenant ID", value="default_tenant", help="Enter your tenant/namespace identifier")
        document_type = st.selectbox("Document Type", ["invoice", "receipt", "bill", "statement"], help="Select document category")

    # Check for existing collections first
    collection_name = f"tenant_{tenant_id}_{document_type}"
    existing_collection = None
    collection_exists = False
    
    # Try to load existing collection
    try:
        embedder = HuggingFaceEmbeddings()
        existing_collection = Chroma(
            embedding_function=embedder,
            persist_directory="./chroma_db",
            collection_name=collection_name
        )
        collection_count = existing_collection._collection.count()
        if collection_count > 0:
            collection_exists = True
            st.success(f"📁 **Found existing collection:** {collection_name} ({collection_count} chunks)")
            st.info("✨ **You can query this collection directly without uploading a file!**")
    except Exception:
        collection_exists = False

    if not collection_exists:
        st.info("📄 **No existing collection found.** Please upload a document to create one.")
    
    uploaded_file = st.file_uploader("Upload your invoice PDF (optional if collection exists)", type="pdf", help="Supported formats: PDF invoices, receipts, bills")

    if uploaded_file or collection_exists:
        embedder = HuggingFaceEmbeddings()
        collection_name = f"tenant_{tenant_id}_{document_type}"
        
        # Initialize variables
        documents = []
        db = None
        document_already_exists = False
        file_hash = None
        
        # Generate file hash for duplicate detection (only if file uploaded)
        if uploaded_file:
            import hashlib
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        # Use existing collection if available, otherwise check/create
        if collection_exists and existing_collection:
            db = existing_collection
            # Get all documents from existing collection
            all_docs = db.get()
            documents = [
                Document(
                    page_content=content,
                    metadata=metadata
                )
                for content, metadata in zip(all_docs['documents'], all_docs['metadatas'])
            ]
            
            # If there's an uploaded file, check for duplicates
            if uploaded_file:
                for doc in documents:
                    existing_filename = doc.metadata.get('source_filename', '')
                    existing_hash = doc.metadata.get('file_hash', '')
                    if (existing_filename == uploaded_file.name or 
                        existing_hash == file_hash):
                        document_already_exists = True
                        break
                
                if document_already_exists:
                    st.warning(f"📄 **Document already exists:** {uploaded_file.name} is already in collection {collection_name}")
                else:
                    st.info(f"➕ **Adding new document:** {uploaded_file.name}")
        
        elif uploaded_file:
            # Try to load existing collection for new file
            with st.spinner("🔍 Checking for existing collection..."):
                try:
                    db = Chroma(
                        embedding_function=embedder,
                        persist_directory="./chroma_db",
                        collection_name=collection_name
                    )
                    
                    collection_count = db._collection.count()
                    if collection_count > 0:
                        collection_exists = True
                        all_docs = db.get()
                        documents = [
                            Document(
                                page_content=content,
                                metadata=metadata
                            )
                            for content, metadata in zip(all_docs['documents'], all_docs['metadatas'])
                        ]
                        
                        # Check for duplicates
                        for doc in documents:
                            existing_filename = doc.metadata.get('source_filename', '')
                            existing_hash = doc.metadata.get('file_hash', '')
                            if (existing_filename == uploaded_file.name or 
                                existing_hash == file_hash):
                                document_already_exists = True
                                break
                        
                        if document_already_exists:
                            st.warning(f"📄 **Document already exists:** {uploaded_file.name}")
                        else:
                            st.info(f"📁 **Found existing collection:** {collection_name} ({collection_count} chunks)")
                            st.info(f"➕ **Adding new document:** {uploaded_file.name}")
                    else:
                        collection_exists = False
                except Exception as e:
                    collection_exists = False
        
        # Process new document if there's a file and (collection doesn't exist or document is new)
        if uploaded_file and (not collection_exists or (collection_exists and not document_already_exists)):
            if collection_exists and not document_already_exists:
                st.info("📄 Adding new document to existing collection...")
            else:
                st.info("📄 Processing new document for new collection...")
        
            # Process documents with comprehensive progress tracking
            new_db, new_documents = process_documents_with_progress(uploaded_file, tenant_id, document_type)
            
            if collection_exists and not document_already_exists and db is not None:
                # Add new documents to existing collection
                db.add_documents(new_documents)
                # Combine document lists
                documents.extend(new_documents)
                st.success(f"➕ **Added to collection:** {collection_name} (added {len(new_documents)} chunks, total: {len(documents)})")
            else:
                # New collection created
                db = new_db
                documents = new_documents
                st.success(f"🆕 **Created new collection:** {collection_name} ({len(documents)} chunks)")

        # Ensure we have a valid database connection
        if db is None:
            st.error("❌ No collection available. Please upload a document first.")
            st.stop()

        # Configure hierarchical retrieval strategy
        section_retriever = db.as_retriever(
            search_type="similarity", 
            search_kwargs={
                "k": 10,
                "filter": {
                    "$and": [
                        {"tenant_id": {"$eq": tenant_id}},
                        {"document_type": {"$eq": document_type}},
                        {"chunk_type": {"$eq": "section"}}
                    ]
                }
            }
        )
        
        document_retriever = db.as_retriever(
            search_type="similarity", 
            search_kwargs={
                "k": 2,
                "filter": {
                    "$and": [
                        {"tenant_id": {"$eq": tenant_id}},
                        {"document_type": {"$eq": document_type}},
                        {"chunk_type": {"$eq": "document"}}
                    ]
                }
            }
        )
        
        # Custom hierarchical retrieval function
        def hierarchical_retrieve(query):
            section_docs = section_retriever.get_relevant_documents(query)
            
            parent_docs = []
            for section_doc in section_docs[:3]:
                parent_id = section_doc.metadata.get("parent_id")
                if parent_id:
                    parent_matches = [doc for doc in documents 
                                    if doc.metadata.get("document_id") == parent_id 
                                    and doc.metadata.get("chunk_type") == "document"]
                    parent_docs.extend(parent_matches)
            
            seen_ids = set()
            unique_parent_docs = []
            for doc in parent_docs:
                doc_id = doc.metadata.get("document_id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_parent_docs.append(doc)
            
            combined_docs = section_docs + unique_parent_docs[:2]
            return combined_docs[:8]
        
        retriever = section_retriever

        llm = Ollama(model="deepseek-r1:8b")

        prompt = """You are an expert invoice analyst. Extract and answer questions about invoice data based ONLY on the provided context.

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

        QA_PROMPT = PromptTemplate.from_template(prompt)
        llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
        combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
        qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

        # Count and display chunk information
        doc_chunks = [d for d in documents if d.metadata.get("chunk_type") == "document"]
        section_chunks = [d for d in documents if d.metadata.get("chunk_type") == "section"]
        
        st.write(f"📊 **Hierarchical Processing Complete:**")
        st.write(f"📄 **Document-level chunks:** {len(doc_chunks)} (full context preservation)")
        st.write(f"📝 **Section-level chunks:** {len(section_chunks)} (granular retrieval)")
        st.write(f"🏢 **Tenant:** {tenant_id} | 📄 **Type:** {document_type} | 🗂️ **Collection:** {collection_name}")

        with st.expander("🏗️ View hierarchical chunk structure"):
            st.subheader("📄 Document-Level Chunks")
            for i, doc in enumerate(doc_chunks):
                st.write(f"**Document {i+1}:**")
                st.json({k: v for k, v in doc.metadata.items() if k in ["document_id", "chunk_type", "chunk_level", "page_number", "total_chars"]})
                st.text_area("", doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content, height=80, key=f"doc_chunk_{i}")
                st.write("---")
            
            st.subheader("📝 Section-Level Chunks")
            for i, doc in enumerate(section_chunks):
                st.write(f"**Section {i+1}:**")
                st.json({k: v for k, v in doc.metadata.items() if k in ["section_id", "parent_id", "chunk_type", "chunk_level", "section_index", "total_chars"]})
                st.text_area("", doc.page_content, height=100, key=f"section_chunk_{i}")
                st.write("---")

        st.markdown("### 💡 Quick Questions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📅 Invoice Date & Number"):
                user_input = "What is the invoice number and invoice date?"
            if st.button("💰 Total Amount"):
                user_input = "What is the total amount of this invoice?"

        with col2:
            if st.button("🏢 Vendor Information"):
                user_input = "Who is the vendor/supplier and their contact details?"
            if st.button("📋 Line Items"):
                user_input = "List all line items with quantities and prices"

        with col3:
            if st.button("💳 Payment Terms"):
                user_input = "What are the payment terms and due date?"
            if st.button("🧮 Tax Breakdown"):
                user_input = "Show the tax breakdown and subtotals"

        manual_input = st.text_input("Or ask your own question about the invoice:")

        if manual_input:
            user_input = manual_input

        if 'user_input' in locals() and user_input:
            print(f"[USER_INPUT]: {user_input}")
            
            # Create enhanced progress tracking for query processing
            query_progress_container = st.container()
            with query_progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                detail_text = st.empty()
            
            # Initialize callback handler
            callback_handler = StreamlitProgressCallback(progress_bar, status_text)
            
            # Step 1: Document retrieval with progress
            status_text.text("🔍 Retrieving relevant documents...")
            detail_text.text("Analyzing query and searching vector database...")
            progress_bar.progress(10)
            
            retrieved_docs = hierarchical_retrieve(user_input)
            
            progress_bar.progress(30)
            status_text.text("📋 Preparing context for analysis...")
            detail_text.text(f"Found {len(retrieved_docs)} relevant chunks")
            
            # Display retrieved chunks
            with st.expander("🎯 Retrieved chunks (hierarchical)"):
                for i, doc in enumerate(retrieved_docs):
                    chunk_type = doc.metadata.get("chunk_type", "unknown")
                    chunk_level = doc.metadata.get("chunk_level", "?")
                    st.write(f"**Chunk {i+1} ({chunk_type}, Level {chunk_level}):**")
                    
                    if chunk_type == "document":
                        st.caption(f"📄 Document ID: {doc.metadata.get('document_id', 'N/A')}")
                    else:
                        st.caption(f"📝 Section ID: {doc.metadata.get('section_id', 'N/A')} | Parent: {doc.metadata.get('parent_id', 'N/A')}")
                    
                    st.text_area("", doc.page_content, height=100, key=f"retrieved_{i}")

            # Step 2: LLM processing with callback
            progress_bar.progress(40)
            status_text.text("🤖 Analyzing with AI model...")
            detail_text.text("Processing context and generating response...")
            
            # Run QA with progress callback
            try:
                response = qa(user_input, callbacks=[callback_handler])["result"]
            except Exception as e:
                # Fallback if callbacks cause issues
                progress_bar.progress(80)
                status_text.text("🤖 Generating response...")
                response = qa(user_input)["result"]
                progress_bar.progress(100)
            
            # Completion
            status_text.text("✅ Response ready!")
            detail_text.text("Analysis complete - displaying results")
            progress_bar.progress(100)
            
            # Brief pause before clearing
            time.sleep(1)
            query_progress_container.empty()
            
            # Display response
            st.markdown("### 📋 Response:")
            st.markdown(response)
            
            # Add response metadata
            with st.expander("ℹ️ Response Details"):
                st.write(f"**Query processed:** {user_input}")
                st.write(f"**Chunks analyzed:** {len(retrieved_docs)}")
                st.write(f"**Collection:** {collection_name}")
                st.write(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with tab2:
    st.markdown("### ChromaDB Collections Overview")
    
    # Function to get all collections information
    def get_collections_info():
        collections_info = []
        try:
            import chromadb
            # Connect to ChromaDB
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Get all collections
            collections = chroma_client.list_collections()
            
            for collection in collections:
                try:
                    # Get collection details
                    coll = chroma_client.get_collection(collection.name)
                    count = coll.count()
                    
                    # Get sample metadata if collection has documents
                    sample_docs = None
                    if count > 0:
                        try:
                            sample_docs = coll.get(limit=5)
                        except Exception:
                            sample_docs = None
                    
                    collections_info.append({
                        "name": collection.name,
                        "count": count,
                        "metadata": collection.metadata or {},
                        "sample_docs": sample_docs
                    })
                except Exception as e:
                    collections_info.append({
                        "name": collection.name,
                        "count": "Error",
                        "metadata": {},
                        "sample_docs": None,
                        "error": str(e)
                    })
                    
        except Exception as e:
            st.error(f"Failed to connect to ChromaDB: {str(e)}")
            return []
            
        return collections_info
    
    # Refresh button
    if st.button("🔄 Refresh Collections"):
        st.rerun()
    
    # Get and display collections
    with st.spinner("📊 Loading collections information..."):
        collections_info = get_collections_info()
    
    if not collections_info:
        st.info("📭 No collections found in ChromaDB")
    else:
        st.success(f"📊 Found {len(collections_info)} collection(s)")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        total_documents = sum(info["count"] for info in collections_info if isinstance(info["count"], int))
        tenant_count = len(set(info["name"].split("_")[1] if "_" in info["name"] else "unknown" for info in collections_info))
        
        with col1:
            st.metric("Total Collections", len(collections_info))
        with col2:
            st.metric("Total Documents", total_documents)
        with col3:
            st.metric("Unique Tenants", tenant_count)
        
        # Collections details
        for info in collections_info:
            with st.expander(f"🗂️ {info['name']} ({info['count']} documents)"):
                # Parse collection name for tenant info
                name_parts = info["name"].split("_")
                if len(name_parts) >= 3 and name_parts[0] == "tenant":
                    tenant_id = name_parts[1]
                    doc_type = "_".join(name_parts[2:])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Tenant ID:** {tenant_id}")
                        st.write(f"**Document Type:** {doc_type}")
                    with col2:
                        st.write(f"**Document Count:** {info['count']}")
                        if "error" in info:
                            st.error(f"Error: {info['error']}")
                
                # Show collection metadata
                if info["metadata"]:
                    st.write("**Collection Metadata:**")
                    st.json(info["metadata"])
                
                # Show sample documents metadata
                if info["sample_docs"] and info["sample_docs"]["metadatas"]:
                    st.write("**Sample Document Metadata:**")
                    
                    # Group by document type
                    doc_types = {}
                    source_files = set()
                    
                    for metadata in info["sample_docs"]["metadatas"]:
                        chunk_type = metadata.get("chunk_type", "unknown")
                        source_file = metadata.get("source_filename", "unknown")
                        
                        if chunk_type not in doc_types:
                            doc_types[chunk_type] = 0
                        doc_types[chunk_type] += 1
                        source_files.add(source_file)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Chunk Types:**")
                        for chunk_type, count in doc_types.items():
                            st.write(f"- {chunk_type}: {count}")
                    
                    with col2:
                        st.write("**Source Files:**")
                        for source_file in sorted(source_files):
                            st.write(f"- {source_file}")
                    
                    # Show detailed metadata for first document
                    if st.checkbox(f"Show detailed metadata for {info['name']}", key=f"detail_{info['name']}"):
                        st.write("**First Document Metadata:**")
                        st.json(info["sample_docs"]["metadatas"][0])
                
                # Collection actions
                st.write("**Actions:**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📊 View Stats", key=f"stats_{info['name']}"):
                        st.info("Detailed statistics feature coming soon!")
                
                with col2:
                    if st.button(f"🗑️ Delete Collection", key=f"delete_{info['name']}", type="secondary"):
                        st.warning("⚠️ Deletion feature requires confirmation - not implemented for safety")
        
        # Add export functionality
        st.markdown("### 📤 Export Collections Data")
        if st.button("📄 Export Collections Summary"):
            import json
            export_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_collections": len(collections_info),
                "total_documents": total_documents,
                "collections": [
                    {
                        "name": info["name"],
                        "document_count": info["count"],
                        "metadata": info["metadata"]
                    }
                    for info in collections_info
                ]
            }
            
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"chroma_collections_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )