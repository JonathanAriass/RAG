"""
Utility functions for the RAG Invoice Q&A Assistant
"""
import hashlib
import datetime
import os
from typing import List, Dict, Any, Optional
import streamlit as st


def generate_file_hash(file_content: bytes) -> str:
    """Generate MD5 hash for file content"""
    return hashlib.md5(file_content).hexdigest()


def create_temp_filename(file_hash: str, original_filename: str) -> str:
    """Create a unique temporary filename"""
    clean_filename = original_filename.replace(' ', '_')
    return f"temp_{file_hash[:8]}_{clean_filename}"


def cleanup_temp_file(filename: str) -> None:
    """Safely remove temporary file"""
    try:
        os.remove(filename)
    except OSError:
        pass  # Ignore cleanup errors


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.datetime.now().isoformat()


def parse_collection_name(collection_name: str) -> tuple[str, str]:
    """
    Parse collection name to extract tenant_id and document_type
    Returns: (tenant_id, document_type)
    """
    parts = collection_name.split("_")
    if len(parts) >= 3 and parts[0] == "tenant":
        tenant_id = parts[1]
        document_type = "_".join(parts[2:])
        return tenant_id, document_type
    return "default_tenant", "invoice"


def create_collection_name(tenant_id: str, document_type: str) -> str:
    """Create standardized collection name"""
    return f"tenant_{tenant_id}_{document_type}"


def display_progress_update(
    progress_bar,
    status_text,
    detail_text,
    progress: int,
    status: str,
    detail: str = ""
) -> None:
    """Update progress indicators"""
    progress_bar.progress(progress)
    status_text.text(status)
    if detail:
        detail_text.text(detail)


def format_document_metadata(
    tenant_id: str,
    document_type: str,
    filename: str,
    file_hash: str,
    chunk_type: str,
    chunk_level: int,
    document_id: str,
    parent_id: Optional[str] = None,
    page_number: int = 1,
    section_index: Optional[int] = None,
    subsection_index: Optional[int] = None,
    section_id: Optional[str] = None,
    total_chars: Optional[int] = None
) -> Dict[str, Any]:
    """Create standardized document metadata"""
    metadata = {
        "tenant_id": tenant_id,
        "document_type": document_type,
        "upload_timestamp": get_current_timestamp(),
        "source_filename": filename,
        "file_hash": file_hash,
        "namespace": f"{tenant_id}_{document_type}",
        "chunk_type": chunk_type,
        "chunk_level": chunk_level,
        "document_id": document_id,
        "parent_id": parent_id,
        "page_number": page_number,
    }
    
    if section_index is not None:
        metadata["section_index"] = section_index
    if subsection_index is not None:
        metadata["subsection_index"] = subsection_index
    if section_id is not None:
        metadata["section_id"] = section_id
    if total_chars is not None:
        metadata["total_chars"] = total_chars
    
    return metadata


def show_chunk_info(documents: List, collection_name: str, tenant_id: str, document_type: str) -> None:
    """Display chunk information in the UI"""
    doc_chunks = [d for d in documents if d.metadata.get("chunk_type") == "document"]
    section_chunks = [d for d in documents if d.metadata.get("chunk_type") == "section"]
    
    st.write("📊 **Hierarchical Processing Complete:**")
    st.write(f"📄 **Document-level chunks:** {len(doc_chunks)} (full context preservation)")
    st.write(f"📝 **Section-level chunks:** {len(section_chunks)} (granular retrieval)")
    st.write(f"🏢 **Tenant:** {tenant_id} | 📄 **Type:** {document_type} | 🗂️ **Collection:** {collection_name}")


def display_chunk_structure(documents: List) -> None:
    """Display hierarchical chunk structure in expandable UI"""
    doc_chunks = [d for d in documents if d.metadata.get("chunk_type") == "document"]
    section_chunks = [d for d in documents if d.metadata.get("chunk_type") == "section"]
    
    with st.expander("🏗️ View hierarchical chunk structure"):
        st.subheader("📄 Document-Level Chunks")
        for i, doc in enumerate(doc_chunks):
            st.write(f"**Document {i+1}:**")
            metadata_display = {
                k: v for k, v in doc.metadata.items() 
                if k in ["document_id", "chunk_type", "chunk_level", "page_number", "total_chars"]
            }
            st.json(metadata_display)
            content_preview = (
                doc.page_content[:300] + "..." 
                if len(doc.page_content) > 300 
                else doc.page_content
            )
            st.text_area("", content_preview, height=80, key=f"doc_chunk_{i}")
            st.write("---")
        
        st.subheader("📝 Section-Level Chunks")
        for i, doc in enumerate(section_chunks):
            st.write(f"**Section {i+1}:**")
            metadata_display = {
                k: v for k, v in doc.metadata.items() 
                if k in ["section_id", "parent_id", "chunk_type", "chunk_level", "section_index", "total_chars"]
            }
            st.json(metadata_display)
            st.text_area("", doc.page_content, height=100, key=f"section_chunk_{i}")
            st.write("---")


def display_retrieved_chunks(retrieved_docs: List) -> None:
    """Display retrieved chunks in expandable UI"""
    with st.expander("🎯 Retrieved chunks (hierarchical)"):
        for i, doc in enumerate(retrieved_docs):
            chunk_type = doc.metadata.get("chunk_type", "unknown")
            chunk_level = doc.metadata.get("chunk_level", "?")
            st.write(f"**Chunk {i+1} ({chunk_type}, Level {chunk_level}):**")
            
            if chunk_type == "document":
                st.caption(f"📄 Document ID: {doc.metadata.get('document_id', 'N/A')}")
            else:
                section_id = doc.metadata.get('section_id', 'N/A')
                parent_id = doc.metadata.get('parent_id', 'N/A')
                st.caption(f"📝 Section ID: {section_id} | Parent: {parent_id}")
            
            st.text_area("", doc.page_content, height=100, key=f"retrieved_{i}")


def display_response_details(
    user_input: str, 
    retrieved_docs: List, 
    collection_name: str
) -> None:
    """Display response metadata"""
    with st.expander("ℹ️ Response Details"):
        st.write(f"**Query processed:** {user_input}")
        st.write(f"**Chunks analyzed:** {len(retrieved_docs)}")
        st.write(f"**Collection:** {collection_name}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.write(f"**Timestamp:** {timestamp}")