"""
UI components module for the RAG Invoice Q&A Assistant
Contains reusable Streamlit UI components and layouts
"""
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from langchain.schema import Document

from utils import parse_collection_name, display_chunk_structure, show_chunk_info


class CollectionSelector:
    """Handles collection selection UI"""
    
    def __init__(self, available_collections: List[str]):
        self.available_collections = available_collections
    
    def render(self) -> Tuple[str, str]:
        """
        Render collection selector UI
        Returns: (tenant_id, document_type)
        """
        if self.available_collections:
            st.info(f"📁 **Available collections:** {len(self.available_collections)} collections found")
            
            use_existing = st.checkbox("🔍 **Select from existing collections**", value=True)
            
            if use_existing:
                selected_collection = st.selectbox("Choose collection:", self.available_collections)
                if selected_collection:
                    tenant_id, document_type = parse_collection_name(selected_collection)
                else:
                    tenant_id, document_type = "default_tenant", "invoice"
            else:
                tenant_id, document_type = self._manual_input()
        else:
            tenant_id, document_type = self._manual_input()
        
        return tenant_id, document_type
    
    def _manual_input(self) -> Tuple[str, str]:
        """Manual tenant ID and document type input"""
        tenant_id = st.text_input(
            "Tenant ID", 
            value="default_tenant", 
            help="Enter your tenant/namespace identifier"
        )
        document_type = st.selectbox(
            "Document Type", 
            ["invoice", "receipt", "bill", "statement"], 
            help="Select document category"
        )
        return tenant_id, document_type


class QuickQuestions:
    """Handles quick question buttons UI"""
    
    def render(self) -> Optional[str]:
        """
        Render quick question buttons
        Returns: Selected question or None
        """
        st.markdown("### 💡 Quick Questions")
        col1, col2, col3 = st.columns(3)
        
        user_input = None
        
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
        
        return user_input


class CollectionOverview:
    """Handles collections overview tab UI"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def render(self):
        """Render collections overview interface"""
        st.markdown("### ChromaDB Collections Overview")
        
        # Refresh button
        if st.button("🔄 Refresh Collections"):
            st.rerun()
        
        # Get and display collections
        with st.spinner("📊 Loading collections information..."):
            collections_info = self.db_manager.get_collections_info()
        
        if not collections_info:
            st.info("📭 No collections found in ChromaDB")
        else:
            self._display_collections_summary(collections_info)
            self._display_collections_details(collections_info)
            self._render_export_section(collections_info)
    
    def _display_collections_summary(self, collections_info: List[Dict]):
        """Display collections summary metrics"""
        st.success(f"📊 Found {len(collections_info)} collection(s)")
        
        stats = self.db_manager.get_collection_statistics(collections_info)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Collections", stats["total_collections"])
        with col2:
            st.metric("Total Documents", stats["total_documents"])
        with col3:
            st.metric("Unique Tenants", stats["unique_tenants"])
    
    def _display_collections_details(self, collections_info: List[Dict]):
        """Display detailed collection information"""
        for info in collections_info:
            with st.expander(f"🗂️ {info['name']} ({info['count']} documents)"):
                self._render_collection_details(info)
    
    def _render_collection_details(self, info: Dict):
        """Render details for a single collection"""
        # Parse collection name for tenant info
        if info["name"].startswith("tenant_"):
            tenant_id, doc_type = parse_collection_name(info["name"])
            
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
            self._render_sample_metadata(info)
        
        # Collection actions
        self._render_collection_actions(info)
    
    def _render_sample_metadata(self, info: Dict):
        """Render sample document metadata"""
        st.write("**Sample Document Metadata:**")
        
        analysis = self.db_manager.analyze_collection_metadata(info["sample_docs"])
        
        if analysis:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Chunk Types:**")
                for chunk_type, count in analysis["chunk_types"].items():
                    st.write(f"- {chunk_type}: {count}")
            
            with col2:
                st.write("**Source Files:**")
                for source_file in analysis["source_files"]:
                    st.write(f"- {source_file}")
        
        # Show detailed metadata for first document
        if st.checkbox(f"Show detailed metadata for {info['name']}", key=f"detail_{info['name']}"):
            st.write("**First Document Metadata:**")
            st.json(info["sample_docs"]["metadatas"][0])
    
    def _render_collection_actions(self, info: Dict):
        """Render collection action buttons"""
        st.write("**Actions:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📊 View Stats", key=f"stats_{info['name']}"):
                st.info("Detailed statistics feature coming soon!")
        
        with col2:
            if st.button(f"🗑️ Delete Collection", key=f"delete_{info['name']}", type="secondary"):
                st.warning("⚠️ Deletion feature requires confirmation - not implemented for safety")
    
    def _render_export_section(self, collections_info: List[Dict]):
        """Render export functionality"""
        st.markdown("### 📤 Export Collections Data")
        if st.button("📄 Export Collections Summary"):
            export_data = self.db_manager.export_collections_summary(collections_info)
            
            filename = f"chroma_collections_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=filename,
                mime="application/json"
            )


class QueryInterface:
    """Handles query interface and response display"""
    
    def __init__(self, qa_chain, hierarchical_retrieve, collection_name: str):
        self.qa_chain = qa_chain
        self.hierarchical_retrieve = hierarchical_retrieve
        self.collection_name = collection_name
    
    def render_query_section(self, documents: List[Document], tenant_id: str, document_type: str):
        """Render the query section with chunk info and quick questions"""
        # Display chunk information
        show_chunk_info(documents, self.collection_name, tenant_id, document_type)
        
        # Display chunk structure
        display_chunk_structure(documents)
        
        # Quick questions
        quick_questions = QuickQuestions()
        user_input = quick_questions.render()
        
        # Manual input
        manual_input = st.text_input("Or ask your own question about the invoice:")
        
        if manual_input:
            user_input = manual_input
        
        if user_input:
            self._process_query(user_input)
    
    def _process_query(self, user_input: str):
        """Process user query with progress tracking"""
        print(f"[USER_INPUT]: {user_input}")
        
        # Create progress tracking
        query_progress_container = st.container()
        with query_progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            detail_text = st.empty()
        
        # Step 1: Document retrieval
        status_text.text("🔍 Retrieving relevant documents...")
        detail_text.text("Analyzing query and searching vector database...")
        progress_bar.progress(10)
        
        retrieved_docs = self.hierarchical_retrieve(user_input)
        
        progress_bar.progress(30)
        status_text.text("📋 Preparing context for analysis...")
        detail_text.text(f"Found {len(retrieved_docs)} relevant chunks")
        
        # Display retrieved chunks
        self._display_retrieved_chunks(retrieved_docs)
        
        # Step 2: LLM processing
        progress_bar.progress(40)
        status_text.text("🤖 Analyzing with AI model...")
        detail_text.text("Processing context and generating response...")
        
        try:
            from callbacks import StreamlitProgressCallback
            callback_handler = StreamlitProgressCallback(progress_bar, status_text)
            response = self.qa_chain(user_input, callbacks=[callback_handler])["result"]
        except Exception as e:
            # Fallback if callbacks cause issues
            progress_bar.progress(80)
            status_text.text("🤖 Generating response...")
            try:
                response = self.qa_chain(user_input)["result"]
            except Exception as fallback_error:
                st.error(f"Error processing query: {str(fallback_error)}")
                return
            progress_bar.progress(100)
        
        # Completion
        status_text.text("✅ Response ready!")
        detail_text.text("Analysis complete - displaying results")
        progress_bar.progress(100)
        
        # Brief pause before clearing
        import time
        time.sleep(1)
        query_progress_container.empty()
        
        # Display response
        self._display_response(response, user_input, retrieved_docs)
    
    def _display_retrieved_chunks(self, retrieved_docs: List[Document]):
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
    
    def _display_response(self, response: str, user_input: str, retrieved_docs: List[Document]):
        """Display the AI response and metadata"""
        st.markdown("### 📋 Response:")
        st.markdown(response)
        
        # Add response metadata
        with st.expander("ℹ️ Response Details"):
            st.write(f"**Query processed:** {user_input}")
            st.write(f"**Chunks analyzed:** {len(retrieved_docs)}")
            st.write(f"**Collection:** {self.collection_name}")
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.write(f"**Timestamp:** {timestamp}")