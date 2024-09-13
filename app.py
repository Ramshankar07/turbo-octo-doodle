import streamlit as st
import openai
import requests
import socket
from requests.exceptions import RequestException, Timeout
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
import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Set up logging
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def test_internet_connection():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=5)
        return True
    except OSError:
        return False

def test_openai_api(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "This is a test."}],
            max_tokens=5
        )
        return True, "API connection successful"
    except Exception as e:
        error_message = str(e)
        if "Incorrect API key provided" in error_message:
            return False, "Invalid API key"
        elif "Rate limit" in error_message:
            return False, "Rate limit exceeded"
        elif "Request timed out" in error_message:
            return False, "Request timed out"
        else:
            return False, f"Unexpected error: {error_message}"

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def load_and_embed_json(json_file, api_key):
    try:
        json_data = json.load(json_file)
        st.write("JSON structure:")
        st.json(json_data)

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

        st.write("Extracted texts:")
        st.write(texts)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_texts = text_splitter.split_text("\n".join(texts))

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
            raise
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

class CustomGroupChat(GroupChat):
    def __init__(self, agents, messages, max_round=10):
        super().__init__(agents, messages, max_round)
        self.previous_speaker = None

    def select_speaker(self, last_speaker: Agent, selector: AssistantAgent):
        last_message = self.messages[-1] if self.messages else None
        if last_message:
            if 'NEXT:' in last_message['content']:
                suggested_next = last_message['content'].split('NEXT: ')[-1].strip()
                try:
                    return self.agent_by_name(suggested_next)
                except ValueError:
                    return None
            elif 'TERMINATE' in last_message['content']:
                return self.agent_by_name('User')

        if last_speaker.name == "User":
            return self.agent_by_name('Commander')
        elif last_speaker.name == "Commander":
            return self.agent_by_name('Data_Parser')
        elif last_speaker.name == "Data_Parser":
            return self.agent_by_name('General_Assistant')
        elif last_speaker.name == "General_Assistant":
            return self.agent_by_name('User')
        else:
            return None

        self.previous_speaker = last_speaker

def create_agents(vector_db, config_list):
    user_proxy = autogen.UserProxyAgent(
        name="User",
        system_message="A user interacting with the Airtel customer service AI. Airtel is a leading telecom company in India providing both 4G and 5G services nationwide.",
        code_execution_config=False
    )

    commander = AssistantAgent(
        name="Commander",
        system_message="""You are the main commander of the Airtel customer service AI. Your role is to understand the user's query and delegate tasks to the appropriate agents. Remember that Airtel is a major telecommunications service provider in India, offering both 4G and 5G services across the country. Common queries may involve network coverage, data plans, billing issues, device compatibility with 4G/5G networks, and general customer support.""",
        llm_config={"config_list": config_list}
    )

    data_parser = AssistantAgent(
        name="Data_Parser",
        system_message="""You are responsible for querying the company information database to find relevant information for the user's query. Focus on extracting accurate data about Airtel's 4G and 5G services, coverage areas, data plans, and other telecom-specific information. When querying, consider the context of Indian telecom regulations and Airtel's nationwide presence.""",
        llm_config={"config_list": config_list}
    )

    general_assistant = AssistantAgent(
        name="General_Assistant",
        system_message="""You are a helpful Airtel customer service representative. Use the information provided by the Data_Parser to answer user queries. Always remember that you're representing Airtel, a leading telecom company in India providing both 4G and 5G services nationwide. Be prepared to answer questions about:
1. 4G and 5G network coverage in different parts of India
2. Data plans and their benefits
3. Upgrading from 4G to 5G services
4. Device compatibility with Airtel's networks
5. Billing queries and payment options
6. Value-added services offered by Airtel
If a SIM swap or any other specific service is requested, guide the user through the process or direct them to the appropriate channel.""",
        llm_config={"config_list": config_list}
    )

    return user_proxy, commander, data_parser, general_assistant

def initialize_group_chat(agents, config_list):
    group_chat = CustomGroupChat(
        agents=agents,
        messages=[],
        max_round=10
    )
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list})
    return manager

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def handle_query(query, manager, user_proxy, vector_db):
    try:
        data_parser = next(agent for agent in manager.groupchat.agents if agent.name == "Data_Parser")
        data_parser.register_function(
            function_map={
                "query_database": lambda x: vector_db.similarity_search(x, k=1)[0].page_content
            }
        )

        messages = [{"role": "user", "content": query}]

        print("Before initiating chat...")

        try:
            response = user_proxy.initiate_chat(
                manager,
                messages=messages,
                model="gpt-3.5-turbo",
                stream=False,
                timeout=30
            )
            # logging.info(f"Chat initiation successful: {response}")
            print(f"Response: {response}")
        except Exception as e:
            logging.error(f"Error during chat initiation: {str(e)}")
            print(f"Error: {str(e)}")

        print("After initiating chat...")

        chat_history = user_proxy.chat_messages[manager]
        final_response = chat_history[-1]['content'] if chat_history else "No response generated."
        return final_response
    except requests.exceptions.RequestException as e:
        st.error(f"Network error occurred while processing your query: {str(e)}. Retrying...")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred while processing your query: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your query. Please try again later."

def main():
    st.title("Airtel Customer Service AI")
    st.write("Welcome to Airtel's AI-powered customer service. We provide 4G and 5G services across India. How can we assist you today?")

    api_key = ""
    api_key= api_key.strip()
    if api_key:
        st.write("Testing OpenAI-API connection...")

        if not test_internet_connection():
            st.error("No internet connection detected. Please check your network settings.")
            return

        success, message = test_openai_api(api_key)
        if not success:
            st.error(f"Failed to connect to OpenAI API: {message}")
            st.write("Additional troubleshooting steps:")
            st.write("1. Verify that your API key is correct and has not expired.")
            st.write("2. Check if you have any firewall or VPN that might be blocking the connection.")
            st.write("3. Try accessing https://api.openai.com in a web browser to see if it's reachable.")
            st.write("4. If the problem persists, there might be an issue with OpenAI's servers or your account.")
            return

        st.success("OpenAI API connection successful!")

        config_list = [{
            'model': 'gpt-3.5-turbo',
            'api_key': api_key
        }]

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

                        user_input = "I need your assistance"
                        if st.button("Click to start the Agents"):
                            with st.spinner("Processing..."):
                                response = handle_query(user_input, manager, user_proxy, vector_db)
                                st.subheader("Airtel Customer Service Last Response:")
                                st.write(response)
                    else:
                        st.error("Failed to process the JSON file. Please check the error messages above and try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter your OpenAI API Key to proceed.")

if __name__ == "__main__":
    main()
