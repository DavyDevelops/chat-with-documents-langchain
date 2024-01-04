import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import docx2txt
import pinecone

def main():
    st.set_page_config(page_title="Ask your Documents")
    st.header("Ask your Documents 💬")

    # load files from directory
    directory = st.text_input("Enter your directory:")
    file_types = ["pdf", "txt", "docx"]
    all_files = [f for f in os.listdir(directory) if any(f.endswith(ft) for ft in file_types)]

    # extract text from files
    all_text = ""
    for file_name in all_files:
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".pdf"):
            with open(file_path, "rb") as f:
                pdf_reader = PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
        elif file_name.endswith(".txt"):
            with open(file_path, "r") as f:
                text = f.read()
        elif file_name.endswith(".docx"):
            text = docx2txt.process(file_path)
        else:
            st.write(f"Unsupported file type: {file_name}")
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

    # create embeddings with Pinecone
    pinecone.init(api_key="b6eb1ad9-3029-449d-9155-1577bad3d8e5")
    index_name = "combsartificialintelligence"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric="cosine", shards=1)
    indexer = pinecone.Index(index_name=index_name)
    embeddings = {chunk: OpenAIEmbeddings().embed(chunk) for chunk in chunks}
    indexer.upsert(items=embeddings)

    # show user input
    user_question = st.text_input("Ask a question about the uploaded documents:")
    if user_question:
        results = indexer.query(queries=[user_question], top_k=5)
        docs = [chunks[i] for i in results.ids[0]]

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(response)

        st.write(response)

if __name__ == '__main__':
    main()
