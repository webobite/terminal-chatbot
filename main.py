# LLM = Large Language Model
# Algorithm that generate text
# Many Opensourced, many closed source (PaLM, BLOOM, GPT 3.0, LLaMA, GLM, Alpaca, OPT, StableLM, Camel)
# Can be hosted by yourself or others

# Most LLM's follows a completion style of text generation (vast majority)

# Input : I'm a comedian who jokes about taxes. Have you ever noticed how taxes ......
                    # Output: are not fun to pay ---> Traditional LLM
# https://platform.openai.com/playground

# Some LLM;s have been adjusted to use a conversational style of generation
        # ChatGPT Bard Claude
        # Back and forth messaging

# OpenAI Terminology
    # User message
    # System message
    # Assistant message

# LangChain Terminology
    # System Message
    # Human Message
    # AI Message

# ChatGPT doesn't remember your conversation
# You must send the entire message history every time you want to extend a conversation

# Memory is used to store data in chain
    # When you run a chain, the memory recieves the input variables and has the ability to add in additional variables
    # after the model runs, the output variables are sent to memory
        # Memory has a chance to inspect the result and store some part of it

# Kinds of memory of langchain
    # COnversationTokenBufferMemory
    # CombinedMemory
    # ConversationBufferWindowMemory

    # ConversationBufferMemory (going to use this)
        # The memory adds a 'HumanMessage' and 'AIMessage' to the list

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)

# setup memory
# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory("messages.json"),
#     memory_key="messages", 
#     return_messages=True
# )

memory = ConversationSummaryMemory(
    memory_key="messages", 
    return_messages=True,
    llm=chat
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    # Debug tag
    verbose=True
)

while True:
    content = input(">> ")
    result = chain({"content": content})

    print(result["text"])