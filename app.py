from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader, PdfFileReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import docx2txt
<<<<<<< HEAD
from dotenv import load_dotenv, find_dotenv
from pymilvus import connections, DataType, Collection, CollectionSchema, FieldSchema, utility
from io import BytesIO
import openai

# Load the .env file
load_dotenv(find_dotenv())

# Get the OPENAI_API_KEY
openai_api_key = os.getenv("COMBS_OPENAI_API_KEY")
zilliz_endpoint = os.getenv("my_zilliz_endpoint")
zilliz_token = os.getenv("my_zilliz_token")
index_name = os.getenv("my_index_name")
=======
>>>>>>> origin/main

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your Documents")
    st.header("Ask your Documents 💬")

    # upload files
<<<<<<< HEAD
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])
    if uploaded_file is not None:
        try:
            # extract text from files
            all_text = ""
            if uploaded_file.name.endswith(".pdf"):
                pdf_reader = PdfFileReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif uploaded_file.name.endswith(".txt"):
                text = uploaded_file.getvalue().decode()
            elif uploaded_file.name.endswith(".docx"):
                text = docx2txt.process(BytesIO(uploaded_file.getvalue()))
            else:
                st.write(f"Unsupported file type: {uploaded_file.name}")
                return
            all_text += text

            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(all_text)

            # create embeddings with Zilliz
            connections.connect(uri=zilliz_endpoint, token=zilliz_token)
            if index_name not in utility.list_collections():
                collection_schema = CollectionSchema(
                    fields=[
                        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
                    ],
                    description="Collection of document embeddings"
                )
                collection = Collection(name=index_name, schema=collection_schema)
            else:
                collection = Collection(name=index_name)
            embeddings = {chunk: openai.Embedding.create(engine="text-davinci-002", text=chunk)["embedding"] for chunk in chunks}
            collection.insert([list(embeddings.values())])

            # show user input
            user_question = st.text_input("Ask a question about the uploaded documents:")
            if user_question:
                query_embedding = openai.Embedding.create(engine="text-davinci-002", text=user_question)["embedding"]
                results = collection.query(top_k=5, expr=f"L2(embedding, {query_embedding.tolist()}) < 1.0")
                docs = [chunks[i.id] for i in results[0]]

                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(response)

                st.write(response)

        except Exception as e:
            st.write(f"Error reading file: {str(e)}")
    else:
        st.write("Please upload a file.")
=======
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

    # Show user input
    user_question = st.text_input("Ask a question about the uploaded documents:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
    
        # instance
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)
       
        st.write(response)
>>>>>>> origin/main

if __name__ == '__main__':
    main()
