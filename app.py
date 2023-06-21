# Importing the required modules
import os 
import streamlit as st
from streamlit_chat import message
import logging
import openai 
import time
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import pinecone 

# Setting up logging configuration
logger = logging.getLogger("AI_Chatbot")

# Setting up Streamlit page configuration
st.set_page_config(
    page_title="AI Chatbot", layout="wide", initial_sidebar_state="expanded"
)

# Getting the OpenAI API key from Streamlit Secrets
openai_api_key = st.secrets.secrets.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key

# Getting the Pinecone API key and environment from Streamlit Secrets
PINECONE_API_KEY = st.secrets.secrets.PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_ENV = st.secrets.secrets.PINECONE_ENV
os.environ["PINECONE_ENV"] = PINECONE_ENV


# Defining the main function
def main():
    # Displaying the heading of the chatbot
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>🧠 AI Chatbot</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Displaying the description of the chatbot
    st.markdown(
        """
        <div style='text-align: center;'>
            <h4>⚡️ Interacting with customized AI!</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    
def admin():
    # Set the Pinecone index name
    pinecone_index = "aichat"

    # Initialize Pinecone with API key and environment
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    # Check if the Pinecone index exists
    if pinecone_index in pinecone.list_indexes():
        index = pinecone.Index(pinecone_index)
        index_stats_response = index.describe_index_stats()

        # Display the available documents in the index
        st.info(f"The Documents available in index: {list(index_stats_response['namespaces'].keys())}")
    
    # Prompt the user to upload PDF/TXT files
    st.write("Upload PDF/TXT Files:")
    uploaded_files = st.file_uploader("Upload", type=["pdf", "txt"], label_visibility="collapsed")#, accept_multiple_files = True
    
    if uploaded_files is not None:
        # Extract the file extension
        file_extension =  os.path.splitext(uploaded_files.name)[1]

        # Create a temporary file and write the uploaded file content
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_files.read())
        
        # Process the uploaded file based on its extension
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
        elif file_extension == ".txt":
            loader = TextLoader(file_path=tmp_file.name, encoding="utf-8")
            pages = loader.load_and_split()

        # Remove the temporary file
        os.remove(tmp_file.name)

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')

        # Display the uploaded file content
        file_container = st.expander(f"Click here to see your uploaded {uploaded_files.name} file:")
        file_container.write(pages)

        # Display success message
        st.success("Document Loaded Successfully!")

        # Checkbox for the first time document upload
        first_t = st.checkbox('Uploading Document First time.')

        st.write("---")

        # Checkbox for subsequent document uploads
        second_t = st.checkbox('Uploading Document Second time and onwards...')

        if first_t:
            # Delete the existing index if it exists
            if pinecone_index in pinecone.list_indexes():
                pinecone.delete_index(pinecone_index)
            time.sleep(50)
            st.info('Initializing Document Uploading to DB...')

            # Create a new Pinecone index
            pinecone.create_index(
                    name=pinecone_index,
                    metric='cosine',
                    dimension=1536  # 1536 dim of text-embedding-ada-002
                    )
            time.sleep(80)

            # Upload documents to the Pinecone index
            st.write(pages)
            vector_store = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index, namespace=uploaded_files.name)
            
            # Display success message
            st.success("Document Uploaded Successfully!")
        
        elif second_t:
            st.info('Initializing Document Uploading to DB...')

            # Upload documents to the Pinecone index
            vector_store = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index, namespace=uploaded_files.name)
            
            # Display success message
            st.success("Document Uploaded Successfully!")


def chat():
    # Set the model name and Pinecone index name
    model_name = "gpt-3.5-turbo" 
    pinecone_index = "aichat"

    # Set the text field for embeddings
    text_field = "text"

    # Create OpenAI embeddings
    embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')

    # load a Pinecone index
    index = pinecone.Index(pinecone_index)
    db = Pinecone(index, embeddings.embed_query, text_field)
    retriever = db.as_retriever()

    # Enable GPT-4 model selection
    mod = st.sidebar.checkbox('Access GPT-4')
    if mod:
        pas = st.sidebar.text_input("Write access code", type="password")
        if pas == "ongpt":
            MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
            model_name = st.sidebar.selectbox(label="Select Model", options=MODEL_OPTIONS)

    
    # Create ChatOpenAI model and RetrievalQA
    llm = ChatOpenAI(model=model_name) # 'gpt-3.5-turbo',
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff", 
                                     retriever=retriever, 
                                     verbose=True)
    
    # Define the prompt form
    def prompt_form():
            """
            Displays the prompt form
            """
            with st.form(key="my_form", clear_on_submit=True):
                # User input
                user_input = st.text_area(
                    "Query:",
                    placeholder="Ask me your queries...",
                    key="input_",
                    label_visibility="collapsed",
                )

                # Submit button
                submit_button = st.form_submit_button(label="Send")
                
                # Check if the form is ready
                is_ready = submit_button and user_input
            return is_ready, user_input
    
    # Define the conversational chat function
    def conversational_chat(query):
        
        # chain_input = {"question": query}#, "chat_history": st.session_state["history"]}
        # result = chain(chain_input)
        # llm = ChatOpenAI(model=model_name)
        # docs = db.similarity_search(query)
        # qa = load_qa_chain(llm=llm, chain_type="stuff")
        # Run the query through the RetrievalQA model
        result = qa.run(query) #chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result))#["answer"]))
    
        return result   #["answer"]
    
    # Initialize session state variables
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me your queries" + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
    
    # Reset chat button
    res = st.button("Reset Chat")

    # Prompt form input and chat processing
    is_ready, user_input = prompt_form()
    if is_ready:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

    # Display chat messages
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile",)
    
    # Reset chat session state
    if res:
        st.session_state['generated'] = []
        st.session_state['past'] = []


# List of available functions: Home, Chatbot, Admin
functions = [
        "Home",
        "Chatbot",
        "Admin",
    ]

# Display a select box in the sidebar to choose the desired function
selected_function = st.sidebar.selectbox("Select Option", functions)

# Call the main() function if "Home" is selected
if selected_function == "Home":
    main()
# Call the chat() function if "Chatbot" is selected
elif selected_function == "Chatbot":
    chat()
elif selected_function == "Admin":
    # Display a text input box in the sidebar to enter the password
    passw = st.sidebar.text_input("Enter your password: ", type="password")
    # Call the admin() function if the correct password is entered
    if passw == "ai4chat":
        admin()
    
