# LLM-PDF-QA
This is a simple implementation of an LLM Based QA engine over custom PDF Input Data.

This notebook  `Pdf_QA.ipynb` is written to allow streamlit application to run in a Colab notebook.
There is an alternate notebook `Pdf_QA_alt.ipynb`version that runs without streamlit. 

## Technologies used
* Embeddings: Huggingface Embeddings model all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* Vectorstore: FAISS (https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
* LLM: OpenAI's GPT-3 davinci (https://platform.openai.com/docs/models)
* Frontend: Streamlit
* Langchain

## Quickstart (Streamlit from Colab)
* Copy the code into your own Colab Notebook.
* Get your own OpenAI API Key.
(You can get your  own key from here: https://platform.openai.com/account/api-keys)
* Run the entire code
* Copy the endpoint ip. (line 1)
* Go to the link. (line 3)
* Enter the endpoint ip.
* You're at the streamlit application!

### In the streamlit application,
* Input your secret key.
* Upload your PDF.
* Ask your question.

## Quickstart (Alt)
* Copy the code into your own Colab Notebook.
* Get your own OpenAI API Key.
(You can get your  own key from here: https://platform.openai.com/account/api-keys)
* Input your secret key in the notebook.
* Input your question in the notebook.
* Run the entire code
