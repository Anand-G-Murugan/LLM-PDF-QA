# Huggingface Embeddings
# OpenAI LLM
# FAISS Vectorstore


import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF QA")
    st.header("PDF QA")

    # get OpenAI API key
    os.environ["OPENAI_API_KEY"] = st.text_input(
        "Enter your OpenAI sk", type="password")
    name = os.environ["OPENAI_API_KEY"]
    if (name):
        st.write("OpenAI key has been entered!")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text from the pdf
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # define embedding function
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2")  # Embeddings model importedfrom Huggingface
        knowledge_base = FAISS.from_texts(chunks, embedding_function)

        # get user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            # selecting LLM
            llm = OpenAI()  # by default -> GPT-3 davinci

            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,
                                     question=user_question)
                print(cb)

            st.write(response)
            st.write(cb)


if __name__ == '__main__':
    main()
