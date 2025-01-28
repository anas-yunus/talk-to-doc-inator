import os
import streamlit as st
import requests
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.base import Embeddings

from dotenv import load_dotenv
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_CHAT_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_EMBEDDING_ENDPOINT = "https://api.mistral.ai/v1/embeddings"

# Custom Embeddings class for Mistral API
class MistralEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text):
        return self._embed_text(text)

    def _embed_text(self, text):
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "input": [text],
            "model": "mistral-embed",
            "encoding_format": "float"
        }
        response = requests.post(MISTRAL_EMBEDDING_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        embeddings = response.json()["output"][0]  # Assuming the response has an "output" field with embeddings
        return embeddings

# Function to process the web document
def load_and_process_document(url):
    # Load the web page
    from langchain.document_loaders import WebBaseLoader
    loader = WebBaseLoader(url)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Use Mistral embeddings
    embeddings = MistralEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)

    return vectorstore

# Function to query Mistral's chat endpoint
def query_mistral_chat(messages):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0.7,
        "max_tokens": 200,
        "messages": messages,
        "response_format": {"type": "text"}
    }
    response = requests.post(MISTRAL_CHAT_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["data"]  # Assuming the response contains the "output" field

# Streamlit app
def main():
    st.title("Website Chatbot (Mistral API)")

    # Input field for the website URL
    url = st.text_input("Enter the website URL:")

    if url:
        st.write("Scraping and processing the website...")

        # Load and process the document
        vectorstore = load_and_process_document(url)

        st.write("Website processed and stored. You can now ask questions.")

        # Chat interface
        st.subheader("Chat with the Website")
        chat_history = []

        # User input for questions
        user_input = st.text_input("You:")

        if user_input:
            # Prepare chat history for Mistral API
            messages = [{"role": "user", "content": user_input}]
            for user_msg, bot_msg in chat_history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_msg})

            # Get the chatbot's response from Mistral
            result = query_mistral_chat(messages)
            chat_history.append((user_input, result))

            # Display the chatbot's response
            st.write(f"Chatbot: {result}")

if __name__ == "__main__":
    main()
