import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Streamlit UI
st.title("📄 RAG System with DeepSeek R1 & Ollama")

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings()
    # vector = FAISS.from_documents(documents, embedder)
    db = Chroma.from_documents(documents, embedder)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = Ollama(model="deepseek-r1:1.5b")
    # llm = Ollama(model="deepseek-r1:14b")

    prompt = """You are an expert document analyst. Answer the question based ONLY on the provided context from the PDF document.

INSTRUCTIONS:
- Provide accurate, detailed answers using information from the context
- If the answer isn't in the context, clearly state "The information is not available in the provided document"
- Quote relevant sections when possible
- Be concise but comprehensive
- Maintain the original meaning and terminology from the document

Context from PDF:
{context}

Question: {question}

Answer:"""

    QA_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

    st.write(f"📊 **Document processed:** {len(documents)} chunks created")
    
    with st.expander("🔍 View all chunks"):
        for i, doc in enumerate(documents):
            st.write(f"**Chunk {i+1}:**")
            st.text_area("", doc.page_content, height=100, key=f"chunk_{i}")
            st.write("---")

    user_input = st.text_input("Ask a question about your document:")

    if user_input:
        retrieved_docs = retriever.get_relevant_documents(user_input)
        
        st.write("🎯 **Retrieved chunks for your question:**")
        for i, doc in enumerate(retrieved_docs):
            st.write(f"**Chunk {i+1} (similarity match):**")
            st.text_area("", doc.page_content, height=100, key=f"retrieved_{i}")
        
        response = qa(user_input)["result"]
        st.write("**Response:**")
        st.write(response)