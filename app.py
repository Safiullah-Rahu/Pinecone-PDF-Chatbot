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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import PyPDF2
from io import StringIO
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


@st.cache_data
def load_docs(files):
    all_text = []
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text.append(text)
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text.append(text)
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text  
def admin():
    # Set the Pinecone index name
    pinecone_index = "aichat"

    # Initialize Pinecone with API key and environment
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    #namespa = st.text_input("Enter Namespace Name: ")
    exist_name = st.checkbox('Use Existing Namespace to Upload Docs')
    del_name = st.checkbox("Delete a Namespace")
    new_name = st.checkbox("Create New Namespace to Upload Docs")
    if exist_name:
        # Check if the Pinecone index exists
        time.sleep(10)
        if pinecone_index in pinecone.list_indexes():
            index = pinecone.Index(pinecone_index)
            index_stats_response = index.describe_index_stats()
            # Display the available documents in the index
            #st.info(f"The Documents available in index: {list(index_stats_response['namespaces'].keys())}")
            # Define the options for the dropdown list
            options = list(index_stats_response['namespaces'].keys())
            
            # Create a dropdown list
            selected_namespace = st.selectbox("Select a namespace", options)
            st.warning("Use 'Uploading Document Second time and onwards...' button to upload docs in existing namespace!", icon="⚠️")
            selected_namespace = selected_namespace
            # Display the selected value
            st.write("You selected:", selected_namespace)
    if del_name:
        if pinecone_index in pinecone.list_indexes():
            index = pinecone.Index(pinecone_index)
            index_stats_response = index.describe_index_stats()
            options = list(index_stats_response['namespaces'].keys())
            selected_namespace = st.selectbox("Select a namespace to delete", options)
            st.warning("The namespace will be permanently deleted!", icon="⚠️")
            del_ = st.checkbox("Check this to delete Namespace")
            if del_:
                with st.spinner('Deleting Namespace...'):
                    time.sleep(5)
                    index.delete(namespace=selected_namespace, delete_all=True)
                st.success('Successfully Deleted Namespace!')


    if new_name:
        selected_namespace = st.text_input("Enter Namespace Name: (For Private Namespaces use .sec at the end, e.g., testname.sec)")

    # Prompt the user to upload PDF/TXT files
    st.write("Upload PDF/TXT Files:")
    uploaded_files = st.file_uploader("Upload", type=["pdf", "txt"], label_visibility="collapsed", accept_multiple_files = True)
    
    if uploaded_files is not None:
        documents = load_docs(uploaded_files)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(documents)

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')

        # Display the uploaded file content
        file_container = st.expander(f"Click here to see your uploaded content:")
        file_container.write(docs)

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
            vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index, namespace= selected_namespace)
            
            # Display success message
            st.success("Document Uploaded Successfully!")
        
        elif second_t:
            st.info('Initializing Document Uploading to DB...')

            # Upload documents to the Pinecone index
            vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index, namespace= selected_namespace)
            
            # Display success message
            st.success("Document Uploaded Successfully!")



def chat(chat_na):
    # Set the model name and Pinecone index name
    model_name = "gpt-3.5-turbo" 
    pinecone_index = "aichat"

    # Set the text field for embeddings
    text_field = "text"

    # Create OpenAI embeddings
    embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')



    # load a Pinecone index
    index = pinecone.Index(pinecone_index)
    db = Pinecone(index, embeddings.embed_query, text_field, namespace=chat_na)
    retriever = db.as_retriever()
    
    # Enable GPT-4 model selection
    mod = st.sidebar.checkbox('Access GPT-4')
    if mod:
        pas = st.sidebar.text_input("Write access code", type="password")
        if pas == "ongpt":
            MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
            model_name = st.sidebar.selectbox(label="Select Model", options=MODEL_OPTIONS)

    
    # Create ChatOpenAI model and RetrievalQA
    # llm = ChatOpenAI(model=model_name) # 'gpt-3.5-turbo',
    # qa = RetrievalQA.from_chain_type(llm=llm,
    #                                  chain_type="stuff", 
    #                                  retriever=retriever, 
    #                                  verbose=True)
    
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
        llm = ChatOpenAI(model=model_name)
        docs = db.similarity_search(query)
        qa = load_qa_chain(llm=llm, chain_type="stuff")
        # Run the query through the RetrievalQA model
        result = qa.run(input_documents=docs, question=query) #chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result))#["answer"]))
    
        return result   #["answer"]
    
    # Initialize session state variables
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me your queries" + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
    
    
    st.write(f"Selected Namespace Name: {chat_na}")
    # Prompt form input and chat processing
    is_ready, user_input = prompt_form()
    if is_ready:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
    # Reset chat button
    res = st.button("Reset Chat")
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
    st.session_state.chat_namesp = ""
    chat_pass = st.sidebar.text_input("Enter chat password: ", type="password")
    if chat_pass == "chatme":
        pinecone_index = "aichat"
        time.sleep(5)
        if pinecone_index in pinecone.list_indexes():
            index = pinecone.Index(pinecone_index)
            index_stats_response = index.describe_index_stats()
            # Define the options for the dropdown list
            options = list(index_stats_response['namespaces'].keys())

        pri_na = st.sidebar.checkbox("Access Private Namespaces")
        chat_namespace = None

        # Check if private namespaces option is selected
        if pri_na:
            pri_pass = st.sidebar.text_input("Write access code:", type="password")
            if pri_pass == "myns":
                chat_namespace = st.sidebar.selectbox("All Namespaces", options)
                st.session_state.chat_namesp = chat_namespace
            else:
                st.info("Enter the correct access code to use private namespaces!")
        else:
            # Filter the options to exclude strings ending with ".sec"
            filtered_list = [string for string in options if not string.endswith(".sec")]
            
            # Create a dropdown list
            chat_namespace = st.sidebar.selectbox("Select a namespace", filtered_list)
            st.session_state.chat_namesp = chat_namespace

        chat_na = st.session_state.chat_namesp
        chat(chat_na)
elif selected_function == "Admin":
    # Display a text input box in the sidebar to enter the password
    passw = st.sidebar.text_input("Enter your password: ", type="password")
    # Call the admin() function if the correct password is entered
    if passw == "ai4chat":
        admin()
    
