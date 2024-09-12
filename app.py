import streamlit as st
import autogen
from autogen.agentchat.groupchat import GroupChat
from autogen.agentchat.agent import Agent
from autogen.agentchat.assistant_agent import AssistantAgent
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import json
import os
import tempfile
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Function to load and embed JSON content with detailed error reporting
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def load_and_embed_json(json_file, api_key):
    try:
        # Read the JSON file
        json_data = json.load(json_file)
        
        # Log the structure of the JSON data
        st.write("JSON structure:")
        st.json(json_data)
        
        # Convert JSON to text for embedding
        texts = []
        for key, value in json_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    texts.append(f"{key} - {sub_key}: {sub_value}")
            elif isinstance(value, list):
                for item in value:
                    texts.append(f"{key}: {item}")
            else:
                texts.append(f"{key}: {value}")
        
        # Log the extracted texts
        st.write("Extracted texts:")
        st.write(texts)
        
        # Split texts
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_texts = text_splitter.split_text("\n".join(texts))
        
        # Create embeddings
        st.write("Creating embeddings... This may take a moment.")
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            db = Chroma.from_texts(split_texts, embeddings)
            st.success("Embeddings created successfully!")
            return db
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error occurred while creating embeddings: {str(e)}")
            st.write("This might be due to network issues or problems with the OpenAI API.")
            st.write("Please check your internet connection and try again.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred while creating embeddings: {str(e)}")
            return None
        
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {str(e)}")
        st.write("Please ensure your JSON file is properly formatted.")
        return None
    except KeyError as e:
        st.error(f"Missing key in JSON structure: {str(e)}")
        st.write("Please check if your JSON file has the expected structure.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the JSON: {str(e)}")
        return None

# ... [rest of the code remains the same] ...

# Streamlit app
def main():
    st.title("Airtel Customer Service AI")
    st.write("Welcome to Airtel's AI-powered customer service. We provide 4G and 5G services across India. How can we assist you today?")

    # API Key input
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    if api_key:
        # Test OpenAI API connection
        st.write("Testing OpenAI API connection...")
        try:
            test_embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            test_embeddings.embed_query("Test")
            st.success("OpenAI API connection successful!")
        except Exception as e:
            st.error(f"Failed to connect to OpenAI API: {str(e)}")
            st.write("Please check your API key and internet connection.")
            return

        # Create config_list using the provided API key
        config_list = [{
            'model': 'gpt-3.5-turbo',
            'api_key': api_key
        }]

        # JSON file uploader
        uploaded_file = st.file_uploader("Upload the company's information document (JSON)", type="json")
        
        if uploaded_file is not None:
            try:
                with st.spinner("Processing the JSON file..."):
                    vector_db = load_and_embed_json(uploaded_file, api_key)
                if vector_db is not None:
                    user_proxy, commander, data_parser, general_assistant = create_agents(vector_db, config_list)
                    agents = [user_proxy, commander, data_parser, general_assistant]
                    manager = initialize_group_chat(agents, config_list)
                    
                    st.success("Company information processed successfully!")

                    # Query input
                    user_input = st.text_input("What's your question about Airtel's 4G/5G services or any other assistance you need?", key="user_input")
                    if st.button("Ask"):
                        with st.spinner("Processing your inquiry..."):
                            response = handle_query(user_input, manager, user_proxy, vector_db)
                        
                        st.subheader("Airtel Customer Service Response:")
                        st.write(response)
                else:
                    st.error("Failed to process the JSON file. Please check the error messages above and try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter your OpenAI API Key to proceed.")

if __name__ == "__main__":
    main()
