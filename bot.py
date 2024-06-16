import streamlit as st
from pypdf import PdfReader
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_text(file):
    loader = TextLoader(file)
    document = loader.load()
    return document[0].page_content

def split_document(text, chunk_size=200, chunk_overlap=10):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return splitter.create_documents(chunks)

def create_embeddings(documents, model_name='hkunlp/instructor-xl', device='cpu'):
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name, model_kwargs={'device': device})
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def create_chatbot(vector_store, model_name='tiiuae/falcon-7b-instruct', temperature=1, max_length=300, token='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'):
    llm = HuggingFaceHub(repo_id=model_name, model_kwargs={'temperature': temperature, 'max_length': max_length}, huggingfacehub_api_token=token)
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)
    return qa

st.title("Chatbot")
st.write("Load a PDF or TXT file to initiate the conversation.")

uploaded_file = st.file_uploader("Load file", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        document_text = load_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        document_text = load_text(uploaded_file)

    st.write("File uploaded successfully.")
    
    if 'vector_store' not in st.session_state:
        documents = split_document(document_text)
        st.session_state.vector_store = create_embeddings(documents)
        st.session_state.qa = create_chatbot(st.session_state.vector_store)

    st.write("Bot is ready. Start the conversation.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What’s new?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = st.session_state.qa({'query': prompt})
        answer = response['result'].split("Helpful Answer:")[-1]
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.write("Please load a PDF or TXT file to start.")