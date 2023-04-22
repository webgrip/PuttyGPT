from cgitb import text
import os
import sys
import time
import asyncio
import weaviate
import openai

# from retriever import WeaviateHybridSearchRetrieverWrapper
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain import OpenAI, LLMChain
from langchain.chains import RetrievalQA
from langchain.agents import (
    initialize_agent,
    load_tools,
    AgentType,
    ZeroShotAgent,
)

# from langchain.callbacks import AimCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.schema import Document
from langchain.memory import (
    ConversationBufferMemory,
    ReadOnlySharedMemory,
)
from langchain.prompts import PromptTemplate
from text_processing import TextProcessing
from tools import create_tools
from langchain.agents import Tool, AgentExecutor
from langchain.schema import Document

from CustomPromptTemplate import CustomPromptTemplate

from langchain.retrievers import TimeWeightedVectorStoreRetriever

from langchain.tools.human.tool import HumanInputRun

# from weaviate_schema import  ScrapedData


from langchain.agents.agent_toolkits import (
    create_vectorstore_router_agent,
    VectorStoreRouterToolkit,
    VectorStoreInfo,
)

from toddleragi.agents.execution_agent import ExecutionAgent
from toddleragi.agents.prioritzation_agent import PrioritizationAgent
from toddleragi.agents.task_creation_agent import TaskCreationAgent

from toddleragi.components.IContextStorage import ContextStorage, ContextData, WeaviateOptions

from collections import deque



import os
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.vectorstores import Weaviate




class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are a task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "incomplete_tasks",
                "objective",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

class ExecutionChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "You are an AI who performs one task based on the following objective: {objective}."
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)



def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(
        result=result,
        task_description=task_description,
        incomplete_tasks=incomplete_tasks,
        objective=objective,
    )
    
    new_tasks = response.split("\n")
    x= [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


    x 
    print('========================================')
    print('RESPONSE')
    print(x)
    print('==============================================')
    print('========================================')
    
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]

def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    this_task_id: int,
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(
        task_names=task_names, next_task_id=next_task_id, objective=objective
    )
    new_tasks = response.split("\n")
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list

def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:


    print(query)

    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_by_vector(embedding_model.embed_query(query), k=k)
    if not results:
        return []
    print(results)

    sorted_results = sorted(results, key=lambda x: x.metadata, reverse=True)
    print(sorted_results)

    return [str(item.metadata["task"]) for item in sorted_results]


def execute_task(
    vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5
) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)


class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list")


        


        self.add_task({"task_id": 1, "task_name": first_task})




        num_iters = 0
        while True:
            if self.task_list:
                

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"


                print('===========================')
                print(task)
                print('===========================')


                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task['task_name']}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    objective,
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain,
                        this_task_id,
                        list(self.task_list),
                        objective,
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print(
                    "\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m"
                )
                break
        return {}

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        execution_chain = ExecutionChain.from_llm(llm, verbose=verbose)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
            vectorstore=vectorstore,
            **kwargs,
        )


OBJECTIVE = "Write me a script that queries langchain's newest documentation and summarizes everything in a simple plain manner."
#OBJECTIVE = "Read the file contents in the entire solution directory this script is running from, and summarize it in clear, clean language. If something fails, you must keep trying."
INITIAL_TASK = "Decide on what solutions would best serve me to reach my objective. An internet search and a summarization might be a good idea."

TASK_STORAGE_NAME = os.getenv("TASK_STORAGE_NAME", os.getenv("TABLE_NAME", "tasks"))
CONTEXT_STORAGE_TYPE = os.getenv("CONTEXT_STORAGE_TYPE", "weaviate")


WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "")
WEAVIATE_VECTORIZER = os.getenv("WEAVIATE_VECTORIZER", "")

assert WEAVIATE_HOST, "WEAVIATE_HOST is missing from .env"
assert WEAVIATE_VECTORIZER, "WEAVIATE_VECTORIZER is missing from .env"

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
schema = client.schema.get()
print(schema)
client.schema.delete_all()

schema = {
    "classes": [
        {
            "class": "Paragraph",
            "vectorizer": "text2vec-openai",
              "moduleConfig": {
                "text2vec-openai": {
                  "model": "ada",
                  "modelversion": "002",
                  "type": "text"
                }
              },
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-openai": {
                          "skip": False,
                          "vectorizePropertyName": False
                        }
                      },
                    "name": "content",
                },
            ],
        },
    ]
}
client.schema.create(schema)







vectorstore = Weaviate(client, "Paragraph", "content")


vectorstore2 = wrappers.weaviate.weaviate_wrapper(vectorstore);



embedding_model = OpenAIEmbeddings(client=client)


#Can we make this gradients?

longTermMemoryRetriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.0000000000000000000000001, k=1)
midTermMemoryRetriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.000000005, k=1) 
shortTermMemoryRetriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.33, k=1) 

sparseAndDenseRetriever = WeaviateHybridSearchRetriever(
    client,
    index_name="LangChain",
    text_key="text",
    alpha=0.5,
    k=4,
    attributes=[],
)

def x(vectorstore):

    global OBJECTIVE
    
    # explainer = lime.LimeTextExplainer(...)

    tracer = LangChainTracer()
    tracer.load_default_session()
    manager = CallbackManager([StdOutCallbackHandler(), tracer])

    openai = OpenAI(temperature=0.415, callback_manager=manager)

    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)

    template = """Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [HumanInput, Memory, Bash, SearchEngine, SummarizeText, SummarizeDocuments]
        Action Input: what to instruct the AI Action representative.
        Observation: The Agent's response
        (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
        Final Answer: the final answer to the original input question with the right amount of detail

        When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response.

        {chat_history}

        Question: {input}

        {agent_scratchpad}
        
    """

    #Chat history: {chat_history}
    tool_names = [tool.name for tool in tools]

    prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
    {agent_scratchpad}"""
    
    


    #hybridSearchChain = RetrievalQA.from_chain_type(
    #    llm=openai,
    #    chain_type="stuff",
    #    retriever=sparseAndDenseRetriever,
    #    verbose=True,
    #    #search_kwargs={"k": 1},
    #)
    
    
    tools = create_tools(manager=manager)

    #tools.append(
    #    Tool(
    #        name="Make a plan",
    #        func=baby_agi.run,
    #        description="Useful for when you start a new task and need a plan to execute. Alway do this first.",
    #        callback_manager=manager
    #    ),
    #)

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )


    llm_chain = LLMChain(llm=openai, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=[tool.name for tool in tools])
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    baby_agi = BabyAGI.from_llm(
        llm=openai,
        vectorstore=vectorstore,
        task_execution_chain=agent_executor,
        verbose=True, 
        max_iterations=2
    )






    baby_agi({"objective": OBJECTIVE})

    
    #agent = initialize_agent(
    #    tools,
    #    openai, 
    #    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #    callback_manager=manager,
    #    #stop=["\nObservation:"], 
    #    max_execution_time=5,
    #    max_iterations=10,
    #    #early_stopping_method="generate",
    #    # return_intermediate_steps=True,
    #    verbose=True,
    #    allowed_tools=[tool.name for tool in tools]
    #)
    
   # print(agent.run(OBJECTIVE))
    #process_data(result)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


def process_data(data):
    print(data)
    text_processing = TextProcessing()

    # Customize the number of sections based on the mode
    num_sections = {"dry": 0, "minimal": 1, "full": len(data["sections"])}['minimal']

    estimated_total_tokens = 0
    estimated_total_duration = 0
    total_cost = 0
    cost_per_token = 0.0006  # You can adjust this

    for i in range(num_sections):

        print(data["sections"][i]);

        # Create document
        section_text = data["sections"][i]["text"]
        section_title = data["sections"][i]["title"]
        page_content = data["sections"][i]["page_content"]

        document = Document(
            text=text,
            meta={
                "name": section_title,
                "url": data["url"],
                "page_content": page_content
            }
        )

        #document.meta["sentiment"] = text_processing.analyze_sentiment(section_text)

        #document.meta["summary"] = text_processing.summarize_concice(section_text)

        tokens = text_processing.count_tokens(section_text)
        document.meta["tokens"] = tokens

        estimated_duration = (
            len(section_text) / 2000 * 5
        )  # Assumes 2000 tokens per second
        estimated_total_duration += estimated_duration
        estimated_total_tokens += tokens
        cost = tokens * cost_per_token * estimated_duration
        total_cost += cost

        document.meta["cost"] = cost

        print(document)

        retriever.add_documents([document])


if __name__ == "__main__":
    try:
        x(vectorstore)
    except openai.error.RateLimitError:
        print(
            "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
        )
    except openai.error.Timeout:
        print(
            "   *** OpenAI API timeout occured. Waiting 10 seconds and trying again. ***"
        )
    except openai.error.APIError:
        print(
            "   *** OpenAI API error occured. Waiting 10 seconds and trying again. ***"
        )
    except openai.error.APIConnectionError:
        print(
            "   *** OpenAI API connection error occured. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
        )
    except openai.error.InvalidRequestError:
        print(
            "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
        )
    except openai.error.ServiceUnavailableError:
        print(
            "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
        )











# import lime


#async def generate_concurrently(questions):
#    agents = []

#    for _ in questions:
#        manager = CallbackManager([StdOutCallbackHandler()])
#        llm = OpenAI(temperature=0, callback_manager=manager)
#        async_tools = load_tools(
#            ["llm-math", "searxng"],
#            llm=llm,
#            aiosession=aiosession,
#            callback_manager=manager,
#        )
#        create_tools(
#            llm_chain=llm_chain, memory=readonlymemory, callback_manager=manager
#        )
#        agents.append(
#            initialize_agent(
#                async_tools,
#                llm,
#                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                verbose=True,
#                callback_manager=manager,
#            )
#        )
#        tasks = [async_agent.arun(q) for async_agent, q in zip(agents, questions)]
#        await asyncio.gather(*tasks)

#    s = time.perf_counter()
#    await asyncio.run(generate_concurrently())
#    elapsed = time.perf_counter() - s
#    print(f"Concurrent executed in {elapsed:0.2f} seconds.")


#def generate_serially(questions):
#    for q in questions:
#        tools = load_tools(["searxng"], llm_chain=llm_chain)
#        agent = initialize_agent(
#            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
#        )
#        agent.run(q)


#def findOpenApiDocumentationUrl(name):
#    return search("does $name have an openapi spec? Give that to me")


#def NLPProcessing():
#    # Slightly tweak the instructions from the default agent
#    openapi_format_instructions = """Use the following format:
#        Question: the input question you must answer
#        Thought: you should always think about what to do
#        Action: the action to take, should be one of [{tool_names}]
#        Action Input: what to instruct the AI Action representative.
#        Observation: The Agent's response
#        (this Thought/Action/Action Input/Observation can repeat N times)
#        Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
#        Final Answer: the final answer to the original input question with the right amount of detail

#        When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response."""

#    speak_toolkit = NLAToolkit.from_llm_and_url(
#        llm, findOpenApiDocumentationUrl("speak"), "https://api.speak.com/openapi.yaml"
#    )
#    klarna_toolkit = NLAToolkit.from_llm_and_url(
#        llm,
#        findOpenApiDocumentationUrl("speak"),
#        "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/",
#    )

#    natural_language_tools = speak_toolkit.get_tools() + klarna_toolkit.get_tools()
#    mrkl = initialize_agent(
#        natural_language_tools,
#        llm,
#        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#        verbose=True,
#        agent_kwargs={"format_instructions": openapi_format_instructions},
#    )


