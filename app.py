import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

 
from docx import Document

def get_file_text(uploaded_files):
    text = ""
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.name.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file.name.endswith(".txt"):
            text += file.read().decode("utf-8") + "\n"
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



def main():
    st.set_page_config(page_title="Chat with Documents", page_icon="📄", layout="wide")

    # Modern dark background with accent color
    st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: #f1f1f1;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3, h4 {
            color: #00C896;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTextInput input {
            background-color: #1e1e1e;
            color: white;
        }
        .stButton > button {
            background-color: #00C896;
            color: black;
            border-radius: 8px;
            font-weight: bold;
        }
        .stFileUploader {
            background-color: #1e1e1e;
            border-radius: 10px;
        }
        .stSpinner {
            color: #00C896;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>📄 Chat with Your Documents</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Upload your PDF, DOCX, or TXT files and ask natural language questions. The answers are AI-powered!</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 2], gap="medium")

    with col1:
        st.header("📂 Upload Files")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button("🚀 Submit & Process"):
            if uploaded_files:
                with st.spinner("Processing your documents..."):
                    raw_text = get_file_text(uploaded_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done! You can now ask questions.")
                    st.balloons()
            else:
                st.warning("⚠️ Please upload at least one file.")

    with col2:
        st.header("💬 Ask a Question")
        st.markdown("Type your question below to get answers based on your uploaded documents.")
        user_question = st.text_input("Ask something...")
        if user_question:
            user_input(user_question)

    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 14px;'>🔒 All processing happens locally. Your files are safe.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 14px;'>💡 Powered by Gemini 1.5 + LangChain + FAISS + Streamlit</p>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
