import streamlit as st
import glob
import os
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory


def get_prompt(instruction, new_system_prompt, B_SYS, E_SYS, B_INST, E_INST):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template
# def clear_llm_memory():
#   bot.memory.clear()

def create_pipeline(max_new_tokens, model, tokenizer):
    pipe = pipeline("text-generation",
                model=model,
                tokenizer = tokenizer,
                max_new_tokens =max_new_tokens,
                temperature =0)
    return pipe
class ChessBot:
  def __init__(self, memory, prompt, retriever, task:str = "text-generation"):
    self.memory = memory
    self.prompt = prompt
    self.retriever = retriever

  def create_chat_bot(self, model, tokenizer, max_new_tokens=1024):
    hf_pipe = create_pipeline(max_new_tokens=max_new_tokens, model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline =hf_pipe)
    qa = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=self.retriever,
      memory=self.memory,
      combine_docs_chain_kwargs={"prompt": self.prompt}
  )
    return qa

def clear_llm_memory(bot):
    bot.memory.clear()
st.title("Chat with pdf using Llama2 🦙🦜")
uploaded_file = st.sidebar.file_uploader("Upload your Data", type="pdf")
if uploaded_file:
    with open(f"./Data/{uploaded_file.name}", "wb+") as f:
        f.write(uploaded_file.getbuffer())

    # bnb_config = BitsAndBytesConfig(load_in_4bit=True,
    #                                 bnb_4bit_quant_type="nf4",
    #                                 bnb_4bit_compute_dtype=torch.bfloat16,
    #                                 bnb_4bit_use_double_quant=False)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model_id = "/home/ptgml/Desktop/lalam2/llama-2-7b-chat-hf/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
    loader = PyPDFLoader(f"./Data/{uploaded_file.name}")

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    pages = loader.load_and_split(text_splitter)
    db = Chroma.from_documents(pages, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"), persist_directory='/home/ptgml/Desktop/lalam2/db')
    instruction = "Given the context that has been provided. \n {context}, Answer the following question - \n{question}"

    system_prompt = """You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."""
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    #get_prompt(instruction=instruction, new_system_prompt=system_prompt, B_SYS=B_SYS, E_SYS=E_SYS, B_INST=B_INST, E_INST=E_INST)
    template = get_prompt(instruction=instruction, new_system_prompt=system_prompt, B_SYS=B_SYS, E_SYS=E_SYS, B_INST=B_INST, E_INST=E_INST)
    print(template)

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5,
        return_messages=True
    )
    retriever = db.as_retriever()
    chess_bot = ChessBot(memory=memory, prompt=prompt, retriever=retriever)
    bot = chess_bot.create_chat_bot(model=model, tokenizer=tokenizer)

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # container for the chat history
    response_container = st.container()
    # container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your pdf data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            clear = st.form_submit_button(label="clear chat")
        if clear:
            clear_llm_memory(bot=bot)
            # import pdb; pdb.set_trace();
            if st.session_state['past']:
                print("i'm clearing session past")
                st.session_state['past'].clear()
            if st.session_state['generated']:
                print("i'm clearing session generated")
                st.session_state['generated'].clear()
        if submit_button and user_input:
            output = bot_message = bot({"question": user_input})['answer']
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    if st.session_state['generated']:
         with response_container:
             for i in range(0,len(st.session_state['generated'])):
                 st.write(str(st.session_state["past"][i]))
                 print(st.session_state["generated"][i])
                 st.write(str(st.session_state["generated"][i]))

else:
    print("i'm clearing")
    files = glob.glob(f"./Data/*")
    files_db = glob.glob(f"./db/index/*")
    for f_1 in files:
        os.remove(f_1)
    for g_1 in files_db:
        os.remove(g_1)

