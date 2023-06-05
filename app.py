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
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import pinecone 


logger = logging.getLogger("AI_Chatbot")
st.set_page_config(
    page_title="AI Chatbot", layout="wide", initial_sidebar_state="expanded"
)


def authenticate(
    openai_api_key: str, PINECONE_API_KEY: str, PINECONE_ENV: str
) -> None:
    # Validate all credentials are set and correct
    # Check for env variables to enable local dev and deployments with shared credentials
    openai_api_key = (
        openai_api_key
        or os.environ.get("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY")
    )
    os.environ["OPENAI_API_KEY"] = openai_api_key
    PINECONE_API_KEY = (
        PINECONE_API_KEY
        or os.environ.get("PINECONE_API_KEY")
        or st.secrets.get("PINECONE_API_KEY")
    )
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    PINECONE_ENV = (
        PINECONE_ENV
        or os.environ.get("PINECONE_ENV")
        or st.secrets.get("PINECONE_ENV")
    )
    os.environ["PINECONE_ENV"] = PINECONE_ENV
    if not (openai_api_key and PINECONE_API_KEY and PINECONE_ENV):
        st.session_state["auth_ok"] = False
        st.error("Credentials neither set nor stored")#, icon=PAGE_ICON)
        return
    try:
        # Try to access openai and deeplake
        with st.spinner("Authentifying..."):
            openai.api_key = openai_api_key
    except Exception as e:
        logger.error(f"Authentication failed with {e}")
        st.session_state["auth_ok"] = False
        st.error("Authentication failed")#, icon=PAGE_ICON)
        return
    # store credentials in the session state
    st.session_state["auth_ok"] = True
    st.session_state["openai_api_key"] = openai_api_key
    st.session_state["PINECONE_API_KEY"] = PINECONE_API_KEY
    st.session_state["PINECONE_ENV"] = PINECONE_ENV
    logger.info("Authentification successful!")
    st.sidebar.success("Authentification successful!")
    # return True

# # Page options and header
# st.set_option("client.showErrorDetails", True)


def main():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>🧠 AI Chatbot</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='text-align: center;'>
            <h4>⚡️ Interacting with customized AI!</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    
def admin():
    pinecone_index = "aichat"
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_description = pinecone.describe_index(pinecone_index)
    st.info(index_description)
    uploaded_files = st.file_uploader("Upload", type=["pdf"], label_visibility="collapsed")#, accept_multiple_files = True
    st.write(uploaded_files.name)
    if uploaded_files is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_files.read())
        loader = PyPDFLoader(tmp_file.name)
        pages = loader.load_and_split()
        os.remove(tmp_file.name)
        embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
        st.success("Document Loaded Successfully!")
        first_t = st.checkbox('Uploading Document First time.')
        st.write("---")
        second_t = st.checkbox('Uploading Document Second time and onwards...')
        if first_t:
            if pinecone_index in pinecone.list_indexes():
                pinecone.delete_index(pinecone_index)
            time.sleep(50)
            st.info('Initializing Document Uploading to DB...')
            pinecone.create_index(
                    name=pinecone_index,
                    metric='cosine',
                    dimension=1536  # 1536 dim of text-embedding-ada-002
                    )
            time.sleep(50)
            vector_store = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index)
            st.success("Document Uploaded Successfully!")
        elif second_t:
            st.info('Initializing Document Uploading to DB...')
            vector_store = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index)
            st.success("Document Uploaded Successfully!")


def chat():
    pinecone_index = "aichat"
    text_field = "text"
    embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
    index = pinecone.Index(pinecone_index)
    db = Pinecone(index, embeddings.embed_query, text_field)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model='gpt-3.5-turbo') # 'gpt-3.5-turbo',
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff", 
                                     retriever=retriever, 
                                     verbose=True)

    def prompt_form():
            """
            Displays the prompt form
            """
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Query:",
                    placeholder="Ask me your queries...",
                    key="input_",
                    label_visibility="collapsed",
                )
                submit_button = st.form_submit_button(label="Send")
                
                is_ready = submit_button and user_input
            return is_ready, user_input
    def conversational_chat(query):
        
        # chain_input = {"question": query}#, "chat_history": st.session_state["history"]}
        # result = chain(chain_input)
        result = qa.run(query) #chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result))#["answer"]))
    
        return result#["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me your queries" + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    res = st.button("Reset Chat")
    is_ready, user_input = prompt_form()
    if is_ready:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    
    if res:
        st.session_state['generated'] = []
        st.session_state['past'] = []

with st.sidebar:
    st.title("Authenticating Credentials")
    with st.form("authentication"):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            #help=OPENAI_HELP,
            placeholder="This field is mandatory",
        )
        PINECONE_API_KEY = st.text_input(
            "Pinecone API Key",
            type="password",
            #help=ACTIVELOOP_HELP,
            placeholder="This field is mandatory",
        )
        PINECONE_ENV = st.text_input(
            "Pinecone Env",
            type="password",
            #help=ACTIVELOOP_HELP,
            placeholder="This field is mandatory",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            authenticate(openai_api_key, PINECONE_API_KEY, PINECONE_ENV)

os.environ["OPENAI_API_KEY"] = openai_api_key

functions = [
        "Home",
        "AI Chatbot",
        "Admin",
    ]



selected_function = st.sidebar.selectbox("Select Option", functions)
if selected_function == "Home":
    main()
elif selected_function == "AI Chatbot":
    chat()
elif selected_function == "Admin":
    passw = st.sidebar.text_input("Enter your password: ", type="password")
    if passw == "ai4chat":
        admin()
    
