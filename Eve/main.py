import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.experimental import BabyAGI

from langchain.callbacks.base import CallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.schema import Document
from langchain.memory import (
    ConversationBufferMemory,
    ReadOnlySharedMemory,
)
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, LLMChain


from langchain.vectorstores import Weaviate
from langchain.retrievers import TimeWeightedVectorStoreRetriever

from agents.AutonomousAgent import AutonomousAgent

from tools import create_tools

from langchain.chat_models import ChatOpenAI

import os

import weaviate 


WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "")
WEAVIATE_VECTORIZER = os.getenv("WEAVIATE_VECTORIZER", "")



# explainer = lime.LimeTextExplainer(...)

tracer = LangChainTracer()
tracer.load_default_session()
callback_manager = CallbackManager([StdOutCallbackHandler(), tracer])

openai = OpenAI(temperature=0.415, callback_manager=callback_manager)

memory = ConversationBufferMemory(memory_key="chat_history")
readonlymemory = ReadOnlySharedMemory(memory=memory)


client = weaviate.Client(
    url=WEAVIATE_HOST,
    additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    # auth_client_secret: Optional[AuthCredentials] = None,
    # timeout_config: Union[Tuple[Real, Real], Real] = (10, 60),
    # proxies: Union[dict, str, None] = None,
    # trust_env: bool = False,
    # additional_headers: Optional[dict] = None,
    # startup_period: Optional[int] = 5,
    # embedded_options=[],
)

embeddings_model = OpenAIEmbeddings(
    #deployment="your-embeddings-deployment-name",
    model="text-embedding-ada-002"
)



def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)

vectorstore = Weaviate(client, "Paragraph", "content", embedding=embeddings_model, relevance_score_fn=relevance_score_fn)

retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)


llm = ChatOpenAI(temperature=0.415, max_tokens=1500, streaming = True, callback_manager=callback_manager) 

autonomousAgent = AutonomousAgent().make(
    name="Ryan",
    age=28,
    traits="loyal, experimental, hopeful, smart, world class programmer",
    status="Executing the task",
    reflection_threshold = 8,
    llm=llm,
    daily_summaries = [
        "Just woke up, ready and eager to start working"
    ],
    verbose=True,
)
##### IDEA: Make a prompt, and let this thing generate the descriptions of what it is and what it's doing, still keep the {objective}
#### TODO Playwright


## THIS IS A TOOL
todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)


tools = create_tools(manager=callback_manager)

tools.append(
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
)


prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)

llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)



OBJECTIVE = "Learn everything there is to learn about langchain."

# Logging of LLMChains
verbose = True
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=verbose, max_iterations=max_iterations
)

baby_agi({"objective": OBJECTIVE})