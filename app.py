"""
Main Streamlit application for the RAG Invoice Q&A Assistant
Refactored version with modular architecture
"""
import streamlit as st

# Set up basic page first
st.title("🧾 Invoice Q&A Assistant")
st.markdown("Upload an invoice PDF and ask questions about vendor details, amounts, dates, line items, and more.")

try:
    from langchain_community.llms import Ollama
    from langchain.chains.llm import LLMChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.chains import RetrievalQA

    # Import custom modules
    from config import AppConfig, PromptTemplates, get_app_title, get_app_description, get_tab_names
    from database_manager import DatabaseManager
    from document_processor import DocumentProcessor
    from ui_components import CollectionSelector, CollectionOverview, QueryInterface
    from utils import create_collection_name
    
    st.success("✅ All modules loaded successfully!")
    modules_loaded = True
except Exception as e:
    st.error(f"❌ Module import failed: {e}")
    st.info("🔧 Please ensure all required packages are installed")
    modules_loaded = False

if modules_loaded:
    class InvoiceQAApp:
        """Main application class for Invoice Q&A Assistant"""
        
        def __init__(self):
            try:
                self.db_manager = DatabaseManager(AppConfig.CHROMA_PERSIST_DIR)
                self.doc_processor = DocumentProcessor()
                self.llm = Ollama(model=AppConfig.DEFAULT_MODEL)
                
                # Setup QA chain
                qa_prompt = PromptTemplates.get_qa_prompt()
                llm_chain = LLMChain(llm=self.llm, prompt=qa_prompt)
                self.combine_documents_chain = StuffDocumentsChain(
                    llm_chain=llm_chain, 
                    document_variable_name="context"
                )
                st.success("✅ App initialized successfully!")
            except Exception as e:
                st.error(f"❌ App initialization failed: {e}")
                raise
        
        def run(self):
            """Run the main application"""
            try:
                self._render_main_interface()
            except Exception as e:
                st.error(f"❌ Error running main interface: {e}")
                st.exception(e)
        
        def _render_main_interface(self):
            """Render the main application interface"""
            tab1_name, tab2_name = get_tab_names()
            tab1, tab2 = st.tabs([tab1_name, tab2_name])
            
            with tab1:
                self._render_document_qa_tab()
            
            with tab2:
                self._render_collections_overview_tab()
        
        def _render_document_qa_tab(self):
            """Render the Document Q&A tab"""
            st.markdown("### Document Q&A Interface")

            try:
                # Get available collections and render selector
                available_collections = self.db_manager.get_available_collections()
                collection_selector = CollectionSelector(available_collections)
                tenant_id, document_type = collection_selector.render()

                # Check for existing collection
                collection_name = create_collection_name(tenant_id, document_type)
                existing_db, collection_exists, documents = self.db_manager.load_existing_collection(
                    tenant_id, document_type
                )

                if collection_exists:
                    st.success(f"📁 **Found existing collection:** {collection_name} ({len(documents)} chunks)")
                    st.info("✨ **You can query this collection directly without uploading a file!**")
                else:
                    st.info("📄 **No existing collection found.** Please upload a document to create one.")
                
                # File uploader
                uploaded_file = st.file_uploader(
                    "Upload your invoice PDF (optional if collection exists)", 
                    type="pdf", 
                    help="Supported formats: PDF invoices, receipts, bills"
                )
                
                # Process document or use existing collection
                if uploaded_file or collection_exists:
                    db, documents = self._handle_document_processing(
                        uploaded_file, existing_db, collection_exists, documents,
                        tenant_id, document_type, collection_name
                    )
                    
                    if db is not None:
                        self._setup_query_interface(db, documents, tenant_id, document_type, collection_name)
            except Exception as e:
                st.error(f"❌ Error in Document Q&A tab: {e}")
                st.exception(e)
        
        def _handle_document_processing(
            self, uploaded_file, existing_db, collection_exists, documents, 
            tenant_id, document_type, collection_name
        ):
            """Handle document processing logic"""
            try:
                db = existing_db
                
                # Check if we need to process a new document
                if uploaded_file:
                    # Check for duplicates if collection exists
                    document_already_exists = False
                    if collection_exists and documents:
                        from utils import generate_file_hash
                        file_hash = generate_file_hash(uploaded_file.getvalue())
                        document_already_exists = self.db_manager.check_document_exists(
                            documents, uploaded_file.name, file_hash
                        )
                    
                    if not collection_exists or not document_already_exists:
                        # Process new document
                        if collection_exists and not document_already_exists:
                            st.info("📄 Adding new document to existing collection...")
                        else:
                            st.info("📄 Processing new document for new collection...")
                        
                        new_db, new_documents = self.doc_processor.process_documents_with_progress(
                            uploaded_file, tenant_id, document_type
                        )
                        
                        if collection_exists and not document_already_exists and db is not None:
                            # Add new documents to existing collection
                            db.add_documents(new_documents)
                            documents.extend(new_documents)
                            st.success(f"➕ **Added to collection:** {collection_name} "
                                     f"(added {len(new_documents)} chunks, total: {len(documents)})")
                        else:
                            # New collection created
                            db = new_db
                            documents = new_documents
                            st.success(f"🆕 **Created new collection:** {collection_name} "
                                     f"({len(documents)} chunks)")
                    else:
                        st.warning(f"📄 **Document already exists:** {uploaded_file.name} "
                                  f"is already in collection {collection_name}")
                
                return db, documents
            except Exception as e:
                st.error(f"❌ Error in document processing: {e}")
                st.exception(e)
                return None, []
        
        def _setup_query_interface(self, db, documents, tenant_id, document_type, collection_name):
            """Setup the query interface"""
            try:
                # Ensure we have a valid database connection
                if db is None:
                    st.error("❌ No collection available. Please upload a document first.")
                    return
                
                # Create retrieval system
                section_retriever, hierarchical_retrieve = self.db_manager.create_hierarchical_retriever(
                    db, tenant_id, document_type, documents
                )
                
                # Create QA chain
                qa = RetrievalQA(
                    combine_documents_chain=self.combine_documents_chain, 
                    retriever=section_retriever
                )
                
                # Create and render query interface
                query_interface = QueryInterface(qa, hierarchical_retrieve, collection_name)
                query_interface.render_query_section(documents, tenant_id, document_type)
            except Exception as e:
                st.error(f"❌ Error setting up query interface: {e}")
                st.exception(e)
        
        def _render_collections_overview_tab(self):
            """Render the Collections Overview tab"""
            try:
                collection_overview = CollectionOverview(self.db_manager)
                collection_overview.render()
            except Exception as e:
                st.error(f"❌ Error in Collections Overview: {e}")
                st.exception(e)

    def main():
        """Main entry point"""
        try:
            app = InvoiceQAApp()
            app.run()
        except Exception as e:
            st.error(f"❌ Failed to start application: {e}")
            st.exception(e)

    if __name__ == "__main__":
        main()

else:
    st.warning("⚠️ Application modules could not be loaded.")
    st.info("💡 The app cannot run without the required dependencies.")
    
    with st.expander("📋 Installation Instructions"):
        st.code("""
# Install required packages
pip install streamlit langchain langchain-community langchain-experimental
pip install chromadb transformers sentence-transformers
pip install pypdf pdfplumber

# Install Ollama and download model
# Visit https://ollama.ai/ for Ollama installation
ollama pull deepseek-r1:8b
        """)