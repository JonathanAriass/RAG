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

# Streamlit UI
st.title("🧾 Invoice Q&A Assistant")
st.markdown("Upload an invoice PDF and ask questions about vendor details, amounts, dates, line items, and more.")

# Tenant/Namespace configuration for multitenant app
tenant_id = st.text_input("Tenant ID", value="default_tenant", help="Enter your tenant/namespace identifier")
document_type = st.selectbox("Document Type", ["invoice", "receipt", "bill", "statement"], help="Select document category")

uploaded_file = st.file_uploader("Upload your invoice PDF", type="pdf", help="Supported formats: PDF invoices, receipts, bills")


if uploaded_file:
    embedder = HuggingFaceEmbeddings()
    # Use tenant-specific collection for namespace isolation
    collection_name = f"tenant_{tenant_id}_{document_type}"
    
    # Check if collection already exists and load it, otherwise create new
    collection_exists = False
    documents = []
    
    try:
        # Try to load existing collection
        db = Chroma(
            embedding_function=embedder,
            persist_directory="./chroma_db",
            collection_name=collection_name
        )
        
        # Check if collection has data
        collection_count = db._collection.count()
        if collection_count > 0:
            collection_exists = True
            st.info(f"📁 **Loaded existing collection:** {collection_name} ({collection_count} chunks)")
            
            # Get all documents from existing collection for hierarchical retrieval
            all_docs = db.get()
            documents = [
                Document(
                    page_content=content,
                    metadata=metadata
                )
                for content, metadata in zip(all_docs['documents'], all_docs['metadatas'])
            ]
        else:
            # Collection exists but is empty, recreate it
            collection_exists = False
            
    except Exception as e:
        # Collection doesn't exist or there's an error, create new one
        collection_exists = False
    
    if not collection_exists:
        # Process the uploaded file and create hierarchical chunks
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PDFPlumberLoader("temp.pdf")
        docs = loader.load()

        # Hierarchical Chunking Strategy
        import datetime
        import uuid
        
        hierarchical_documents = []
        
        # Level 1: Document-level chunks (preserve full document context)
        for doc_idx, doc in enumerate(docs):
            doc_id = str(uuid.uuid4())
            
            # Create document-level chunk
            doc_chunk = Document(
                page_content=doc.page_content,
                metadata={
                    "tenant_id": tenant_id,
                    "document_type": document_type,
                    "upload_timestamp": datetime.datetime.now().isoformat(),
                    "source_filename": uploaded_file.name,
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
            
            # Level 2: Section-level chunks (granular retrieval)
            # Use semantic chunker for intelligent splitting
            semantic_splitter = SemanticChunker(
                HuggingFaceEmbeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=75
            )
            section_chunks = semantic_splitter.split_documents([doc])
            
            # If semantic chunker produces large chunks, further split with recursive splitter
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
                                "upload_timestamp": datetime.datetime.now().isoformat(),
                                "source_filename": uploaded_file.name,
                                "namespace": f"{tenant_id}_{document_type}",
                                "chunk_type": "section",
                                "chunk_level": 2,
                                "document_id": doc_id,
                                "parent_id": doc_id,
                                "section_id": f"{doc_id}_s{section_idx}_sub{subsection_idx}",
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
                            "upload_timestamp": datetime.datetime.now().isoformat(),
                            "source_filename": uploaded_file.name,
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
        
        documents = hierarchical_documents
        
        # Create new collection with documents
        db = Chroma.from_documents(
            documents,
            embedder,
            persist_directory="./chroma_db",
            collection_name=collection_name
        )
        st.success(f"🆕 **Created new collection:** {collection_name} ({len(documents)} chunks)")

    # Configure hierarchical retrieval strategy
    # Primary retriever for section-level chunks (granular search)
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
    
    # Secondary retriever for document-level chunks (context preservation)
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
        # Get relevant sections
        section_docs = section_retriever.get_relevant_documents(query)
        
        # Get parent documents for context
        parent_docs = []
        for section_doc in section_docs[:3]:  # Top 3 sections
            parent_id = section_doc.metadata.get("parent_id")
            if parent_id:
                parent_matches = [doc for doc in documents 
                                if doc.metadata.get("document_id") == parent_id 
                                and doc.metadata.get("chunk_type") == "document"]
                parent_docs.extend(parent_matches)
        
        # Remove duplicates while preserving order
        seen_ids = set()
        unique_parent_docs = []
        for doc in parent_docs:
            doc_id = doc.metadata.get("document_id")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_parent_docs.append(doc)
        
        # Combine section and document chunks, prioritizing sections
        combined_docs = section_docs + unique_parent_docs[:2]
        return combined_docs[:8]  # Limit total context
    
    # Use section retriever as primary for backward compatibility
    retriever = section_retriever

    # llm = Ollama(model="deepseek-r1:1.5b")
    llm = Ollama(model="deepseek-r1:8b")
    # llm = Ollama(model="deepseek-r1:14b")

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

    # Count chunk types for display
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
        
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Document retrieval
        status_text.text("🔍 Retrieving relevant documents...")
        progress_bar.progress(20)
        retrieved_docs = hierarchical_retrieve(user_input)
        
        # Step 2: Context preparation
        status_text.text("📋 Preparing context for analysis...")
        progress_bar.progress(40)
        
        with st.expander("🎯 Retrieved chunks (hierarchical)"):
            for i, doc in enumerate(retrieved_docs):
                chunk_type = doc.metadata.get("chunk_type", "unknown")
                chunk_level = doc.metadata.get("chunk_level", "?")
                st.write(f"**Chunk {i+1} ({chunk_type}, Level {chunk_level}):**")
                
                # Show relevant metadata
                if chunk_type == "document":
                    st.caption(f"📄 Document ID: {doc.metadata.get('document_id', 'N/A')}")
                else:
                    st.caption(f"📝 Section ID: {doc.metadata.get('section_id', 'N/A')} | Parent: {doc.metadata.get('parent_id', 'N/A')}")
                
                st.text_area("", doc.page_content, height=100, key=f"retrieved_{i}")

        # Step 3: LLM processing
        status_text.text("🤖 Generating response with AI model...")
        progress_bar.progress(60)
        
        # Step 4: Response generation
        status_text.text("✍️ Analyzing invoice data...")
        progress_bar.progress(80)
        response = qa(user_input)["result"]
        
        # Step 5: Complete
        status_text.text("✅ Response ready!")
        progress_bar.progress(100)
        
        # Clear progress indicators after a brief moment
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        st.write("**Response:**")
        st.write(response)
