import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

BASE_DB_DIR = "chroma_dbs"
UPLOAD_DIR = "uploaded_docs"
os.makedirs(BASE_DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="AI Bot", layout="wide")
st.title("📄AI Bot ")

uploaded_file = st.file_uploader("Upload a document (PDF, TXT, or MD)", type=["pdf", "txt", "md"])

if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f" uploaded: {uploaded_file.name}")

    file_db = os.path.splitext(uploaded_file.name)[0] + "_db"
    db_path = os.path.join(BASE_DB_DIR, file_db)

   
    start_time = time.perf_counter()
    if not os.path.exists(db_path):
        st.info(" Loading document")
        loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        
        chunk_size = 500
        chunk_overlap = 100
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = splitter.split_documents(docs)
        
        st.info(f" Number of chunks: {len(split_docs)} (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        st.info(" Creating embeddings (first time only)...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=db_path)
        st.success("Embeddings created")
    else:
        st.info(" Loading embeddings...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
        st.success("Embeddings loaded!")

    db_latency = time.perf_counter() - start_time
    st.write(f"latency: {db_latency:.2f}s")

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    
    llm_model = st.selectbox("Choose LLM", ["mistral", "llama2"])
    llm = OllamaLLM(model=llm_model, options={"num_ctx": 512, "max_tokens": 256})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    st.success("Ask questions below.")

    query = st.text_input("Ask a question :")
    if query:
        start_time = time.perf_counter()
        try:
            with st.spinner("Thinking..."):
                result = qa.invoke(query) 
                answer = result["result"]
                sources = result.get("source_documents", [])
            
            latency = time.perf_counter() - start_time
            st.write(f"latency: {latency:.2f}s")
            
            st.markdown("*Answer:*")
            st.write(answer)

            if sources:
                st.markdown("*Cited Chunks:*")
                for i, doc in enumerate(sources):
                    st.markdown(f"- *Chunk {i+1}:* {doc.page_content[:300]}...") 

        except Exception as e:
            st.error(f"⚠ Error generating answer: {e}")
            st.info("Try increasing k or adjusting chunk size/overlap in ingestion.")

    db_size = sum(os.path.getsize(os.path.join(db_path, f)) for f in os.listdir(db_path))
    st.info(f"💾 Vector DB size: {db_size / 1024:.2f} KB")
