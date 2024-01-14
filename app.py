from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import docx2txt

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your Documents")
    st.header("Ask your Documents 💬")

    # upload files
    uploaded_files = st.file_uploader("Upload your files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    # extract text from uploaded files
    all_text = ""
    for uploaded_file in uploaded_files:

        # File type handling 
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            text_list = []
            for page in pdf_reader.pages:
                text_list.append(page.extract_text())
        elif uploaded_file.name.endswith((".txt", ".text")):
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".docx"):
            text = docx2txt.process(uploaded_file)
        else:
            st.write(f"Unsupported file type: {uploaded_file.name}")
            continue
        all_text += text
    
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)
  
    # Create Embeddings
    embeddings = OpenAIEmbeddings()
    if embeddings:
        knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)  # Update the parameter names
    else:
        st.write("Error: No embedding found")

    # show user input
    user_question = st.text_input("Ask a question about the uploaded documents:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
    
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)
       
        st.write(response)

if __name__ == '__main__':
    main()
