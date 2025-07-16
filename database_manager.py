"""
Database operations module for the RAG Invoice Q&A Assistant
Handles ChromaDB operations and collection management
"""
import datetime
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from utils import create_collection_name, parse_collection_name


class DatabaseManager:
    """Manages ChromaDB operations and collection handling"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedder = HuggingFaceEmbeddings()
    
    def get_available_collections(self) -> List[str]:
        """Get list of available tenant collections"""
        available_collections = []
        try:
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            collections = chroma_client.list_collections()
            for collection in collections:
                if collection.name.startswith("tenant_"):
                    coll = chroma_client.get_collection(collection.name)
                    if coll.count() > 0:
                        available_collections.append(collection.name)
        except Exception:
            pass
        return available_collections
    
    def load_existing_collection(
        self, 
        tenant_id: str, 
        document_type: str
    ) -> Tuple[Optional[Chroma], bool, List[Document]]:
        """
        Load existing collection if it exists
        Returns: (collection, exists, documents)
        """
        collection_name = create_collection_name(tenant_id, document_type)
        documents = []
        
        try:
            db = Chroma(
                embedding_function=self.embedder,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )
            
            collection_count = db._collection.count()
            if collection_count > 0:
                # Get all documents from existing collection
                all_docs = db.get()
                documents = [
                    Document(
                        page_content=content,
                        metadata=metadata
                    )
                    for content, metadata in zip(all_docs['documents'], all_docs['metadatas'])
                ]
                return db, True, documents
            else:
                return None, False, []
        except Exception:
            return None, False, []
    
    def check_document_exists(
        self, 
        documents: List[Document], 
        filename: str, 
        file_hash: str
    ) -> bool:
        """Check if document already exists in collection"""
        for doc in documents:
            existing_filename = doc.metadata.get('source_filename', '')
            existing_hash = doc.metadata.get('file_hash', '')
            if (existing_filename == filename or existing_hash == file_hash):
                return True
        return False
    
    def create_hierarchical_retriever(
        self, 
        db: Chroma, 
        tenant_id: str, 
        document_type: str,
        documents: List[Document]
    ):
        """Create hierarchical retrieval system"""
        # Section retriever
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
        
        # Document retriever
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
        
        def hierarchical_retrieve(query: str) -> List[Document]:
            """Custom hierarchical retrieval function"""
            section_docs = section_retriever.get_relevant_documents(query)
            
            parent_docs = []
            for section_doc in section_docs[:3]:
                parent_id = section_doc.metadata.get("parent_id")
                if parent_id:
                    parent_matches = [
                        doc for doc in documents
                        if (doc.metadata.get("document_id") == parent_id and
                            doc.metadata.get("chunk_type") == "document")
                    ]
                    parent_docs.extend(parent_matches)
            
            # Remove duplicates
            seen_ids = set()
            unique_parent_docs = []
            for doc in parent_docs:
                doc_id = doc.metadata.get("document_id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_parent_docs.append(doc)
            
            combined_docs = section_docs + unique_parent_docs[:2]
            return combined_docs[:8]
        
        return section_retriever, hierarchical_retrieve
    
    def get_collections_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about all collections"""
        collections_info = []
        try:
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            collections = chroma_client.list_collections()
            
            for collection in collections:
                try:
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
    
    def export_collections_summary(self, collections_info: List[Dict]) -> Dict[str, Any]:
        """Create exportable collections summary"""
        total_documents = sum(
            info["count"] for info in collections_info 
            if isinstance(info["count"], int)
        )
        
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
        return export_data
    
    def get_collection_statistics(self, collections_info: List[Dict]) -> Dict[str, int]:
        """Calculate collection statistics"""
        total_documents = sum(
            info["count"] for info in collections_info 
            if isinstance(info["count"], int)
        )
        
        tenant_count = len(set(
            parse_collection_name(info["name"])[0] 
            for info in collections_info 
            if info["name"].startswith("tenant_")
        ))
        
        return {
            "total_collections": len(collections_info),
            "total_documents": total_documents,
            "unique_tenants": tenant_count
        }
    
    def analyze_collection_metadata(self, sample_docs: Dict) -> Dict[str, Any]:
        """Analyze sample document metadata"""
        if not sample_docs or not sample_docs.get("metadatas"):
            return {}
        
        doc_types = {}
        source_files = set()
        
        for metadata in sample_docs["metadatas"]:
            chunk_type = metadata.get("chunk_type", "unknown")
            source_file = metadata.get("source_filename", "unknown")
            
            if chunk_type not in doc_types:
                doc_types[chunk_type] = 0
            doc_types[chunk_type] += 1
            source_files.add(source_file)
        
        return {
            "chunk_types": doc_types,
            "source_files": sorted(source_files)
        }