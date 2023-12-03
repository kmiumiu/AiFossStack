import chromadb
import os

import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import Ollama
from llama_index.evaluation import FaithfulnessEvaluator

st.set_page_config(page_title="Local: Chat with your own content!", layout="wide", initial_sidebar_state="auto", menu_items=None)

with st.sidebar:
    model = st.radio('Model:', ['orca-mini','zephyr'])
    embed_lib = st.radio('Embedding Library:', ['local'])
    knowledgebase = st.text_input("Data Folder:", value="data" )
    chat_mode = st.radio('Chat Mode:', ['condense_question', 'context', 'simple'])
    temperature = st.slider('Temperature:', min_value=0., max_value=1., value=0.5)

st.image("logo.png", width=400)
st.title("Your content + your local AI language model = your privacy.")
system_prompt= st.text_area("Enter the role this AI assitant should play:" , value="You are my expert advisor. Assume that all questions are related to the data folder indicated above. For each fact you respond always include the reference document and page or paragraph. Keep your answers based on facts. Cite the source document next to each paragraph response you provide. Do not hallucinate features.")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question:"}
    ]

service_context = ServiceContext.from_defaults(embed_model=embed_lib, llm=Ollama(model=model, temperature=temperature))  #try model="zephyr" for better but slower results.

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing docs – hang tight! This should take 1-2 minutes."):
        if not os.path.exists("./chroma_db"):
            # load some documents
            documents = SimpleDirectoryReader(input_dir="./data", recursive=True).load_data()

            # initialize client, setting path to save data
            db = chromadb.PersistentClient(path="./chroma_db")

            # create collection
            chroma_collection = db.get_or_create_collection("quickstart")

            # assign chroma as the vector_store to the context
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # create your index
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, service_context=service_context
            )

        else: 
            # initialize client
            db = chromadb.PersistentClient(path="./chroma_db")

            # get collection
            chroma_collection = db.get_or_create_collection("quickstart")

            # assign chroma as the vector_store to the context
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # load your index from stored vectors
            index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context, service_context=service_context
            )

        return index

index = load_data()

# define evaluator
evaluator = FaithfulnessEvaluator(service_context=service_context)

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode=chat_mode, verbose=True) #modes might be "condense_question" or "context" or "simple"

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            eval_result = evaluator.evaluate_response(response=response)
            text = response.response if eval_result.passing else "I do not have enough information"
            st.write(text)
            lenght_sources = len(response.source_nodes)
            with st.expander("Show References"):
                for i in range(lenght_sources):
                    st.write(response.source_nodes[i].metadata)
            
            message = {"role": "assistant", "content": text}
            st.session_state.messages.append(message) # Add response to message history
