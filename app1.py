import streamlit as st
import os
import re
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from tiktoken import get_encoding

# Set up Groq and OpenAI API keys
groq_api_key = "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"
os.environ["OPENAI_API_KEY"] = "enter your api key here"

# Function to clean text by removing special characters
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text)

# Function to check token count and ensure it doesn't exceed model limits
def check_token_count(text, model_name="cl100k_base", max_tokens=4096):
    encoding = get_encoding(model_name)
    token_count = len(encoding.encode(text))
    return token_count <= max_tokens, token_count

# Streamlit app setup
st.title("LangChain Demo with Groq 🐦🔗🐦🔗⚡🖥️⚡🖥️")

# Initialize session state if not already done
if "vectordb" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = WebBaseLoader("https://smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectordb = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

# Initialize the Groq-based LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based on the given context only. 
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Create document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectordb.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

# Get user input
prompt = st.text_input("Write your input prompt here......")

# If a prompt is provided, process it
if prompt:
    # Clean the prompt and check token length
    cleaned_prompt = clean_text(prompt)
    is_within_limit, token_count = check_token_count(cleaned_prompt)

    if not is_within_limit:
        st.write(f"Your input is too long ({token_count} tokens). Please shorten it.")
    else:
        try:
            # Measure response time
            start_time = time.process_time()
            response = retriever_chain.invoke({"input": cleaned_prompt})
            response_time = time.process_time() - start_time
            st.write(f"Response Time: {response_time:.2f} seconds")
            
            # Display the response
            st.write(response['answer'])

            # Display the documents in the expander for similarity search
            with st.expander("Document similarity search"):
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write("------------------------------------")

        except Exception as e:
            st.write("Error processing the prompt. Please check the logs.")
            st.error(f"Exception: {e}")
