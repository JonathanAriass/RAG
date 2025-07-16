"""
Document processing module for the RAG Invoice Q&A Assistant
Handles PDF loading, text extraction, and hierarchical chunking
"""
import uuid
import time
from typing import List, Tuple
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma

from utils import (
    generate_file_hash,
    create_temp_filename,
    cleanup_temp_file,
    format_document_metadata,
    display_progress_update
)


class DocumentProcessor:
    """Handles document processing with progress tracking"""
    
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings()
    
    def process_documents_with_progress(
        self, 
        uploaded_file, 
        tenant_id: str, 
        document_type: str
    ) -> Tuple[Chroma, List[Document]]:
        """Process uploaded PDF with detailed progress tracking"""
        
        # Initialize progress tracking
        progress_container = st.container()
        with progress_container:
            main_progress = st.progress(0)
            status_text = st.empty()
            detail_text = st.empty()
        
        try:
            # Stage 1: File Processing (20%)
            display_progress_update(
                main_progress, status_text, detail_text,
                10, "📄 Loading PDF document..."
            )
            
            # Create unique temporary filename
            file_hash = generate_file_hash(uploaded_file.getvalue())
            temp_filename = create_temp_filename(file_hash, uploaded_file.name)
            
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            loader = PDFPlumberLoader(temp_filename)
            display_progress_update(
                main_progress, status_text, detail_text,
                20, "📖 Extracting text from PDF..."
            )
            docs = loader.load()
            
            # Stage 2: Document Structure Analysis (40%)
            display_progress_update(
                main_progress, status_text, detail_text,
                30, "🏗️ Analyzing document structure..."
            )
            
            hierarchical_documents = []
            total_docs = len(docs)
            
            # Process each document with sub-progress
            for doc_idx, doc in enumerate(docs):
                detail_text.text(f"Processing page {doc_idx + 1} of {total_docs}")
                
                doc_id = str(uuid.uuid4())
                
                # Create document-level chunk
                doc_chunk = Document(
                    page_content=doc.page_content,
                    metadata=format_document_metadata(
                        tenant_id=tenant_id,
                        document_type=document_type,
                        filename=uploaded_file.name,
                        file_hash=file_hash,
                        chunk_type="document",
                        chunk_level=1,
                        document_id=doc_id,
                        parent_id=None,
                        page_number=doc.metadata.get("page", doc_idx + 1),
                        total_chars=len(doc.page_content)
                    )
                )
                hierarchical_documents.append(doc_chunk)
                
                # Update progress for document processing
                doc_progress = 30 + (doc_idx + 1) / total_docs * 20
                main_progress.progress(int(doc_progress))
            
            # Stage 3: Semantic Chunking (60%)
            display_progress_update(
                main_progress, status_text, detail_text,
                50, "🧠 Creating semantic chunks...",
                "Loading embedding model..."
            )
            
            # Process semantic chunking
            hierarchical_documents.extend(
                self._create_semantic_chunks(
                    docs, hierarchical_documents, uploaded_file.name, 
                    file_hash, tenant_id, document_type,
                    main_progress, detail_text
                )
            )
            
            # Stage 4: Vector Store Creation (80%)
            display_progress_update(
                main_progress, status_text, detail_text,
                70, "🗄️ Creating vector database..."
            )
            
            collection_name = f"tenant_{tenant_id}_{document_type}"
            detail_text.text(f"Building collection: {collection_name}")
            
            db = self._create_vector_store(
                hierarchical_documents, collection_name,
                main_progress, detail_text
            )
            
            # Stage 5: Finalization (100%)
            display_progress_update(
                main_progress, status_text, detail_text,
                100, "✅ Processing complete!",
                f"Created {len(hierarchical_documents)} chunks"
            )
            
            # Brief pause to show completion
            time.sleep(1)
            
            # Cleanup
            cleanup_temp_file(temp_filename)
            progress_container.empty()
            
            return db, hierarchical_documents
            
        except Exception as e:
            status_text.text(f"❌ Error during processing: {str(e)}")
            main_progress.progress(0)
            cleanup_temp_file(temp_filename)
            raise e
    
    def _create_semantic_chunks(
        self, 
        docs: List[Document], 
        hierarchical_documents: List[Document],
        filename: str,
        file_hash: str,
        tenant_id: str,
        document_type: str,
        main_progress,
        detail_text
    ) -> List[Document]:
        """Create semantic chunks for documents"""
        semantic_chunks = []
        total_docs = len(docs)
        
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
                            metadata=format_document_metadata(
                                tenant_id=tenant_id,
                                document_type=document_type,
                                filename=filename,
                                file_hash=file_hash,
                                chunk_type="section",
                                chunk_level=2,
                                document_id=doc_id,
                                parent_id=doc_id,
                                page_number=doc.metadata.get("page", doc_idx + 1),
                                section_index=section_idx,
                                subsection_index=subsection_idx,
                                section_id=f"{doc_id}_s{section_idx}_sub{subsection_idx}",
                                total_chars=len(subsection.page_content)
                            )
                        )
                        semantic_chunks.append(section_chunk)
                else:
                    section_chunk = Document(
                        page_content=section.page_content,
                        metadata=format_document_metadata(
                            tenant_id=tenant_id,
                            document_type=document_type,
                            filename=filename,
                            file_hash=file_hash,
                            chunk_type="section",
                            chunk_level=2,
                            document_id=doc_id,
                            parent_id=doc_id,
                            page_number=doc.metadata.get("page", doc_idx + 1),
                            section_index=section_idx,
                            section_id=f"{doc_id}_s{section_idx}",
                            total_chars=len(section.page_content)
                        )
                    )
                    semantic_chunks.append(section_chunk)
            
            # Update progress for semantic processing
            semantic_progress = 50 + (doc_idx + 1) / total_docs * 20
            main_progress.progress(int(semantic_progress))
        
        return semantic_chunks
    
    def _create_vector_store(
        self, 
        documents: List[Document], 
        collection_name: str,
        main_progress,
        detail_text
    ) -> Chroma:
        """Create vector store with batch processing"""
        batch_size = 50
        total_batches = len(documents) // batch_size + (
            1 if len(documents) % batch_size else 0
        )
        
        db = None
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]
            
            detail_text.text(f"Processing batch {batch_idx + 1} of {total_batches}")
            
            if db is None:
                db = Chroma.from_documents(
                    batch_docs,
                    self.embedder,
                    persist_directory="./chroma_db",
                    collection_name=collection_name
                )
            else:
                db.add_documents(batch_docs)
            
            # Update progress for vector store creation
            batch_progress = 70 + (batch_idx + 1) / total_batches * 25
            main_progress.progress(int(batch_progress))
        
        return db