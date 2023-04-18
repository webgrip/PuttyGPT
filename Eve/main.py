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
from searxng_wrapper import SearxNGWrapper
from langchain.schema import Document

from CustomPromptTemplate import CustomPromptTemplate

from langchain.retrievers import TimeWeightedVectorStoreRetriever

# from weaviate_schema import  ScrapedData


#OBJECTIVE = "Write me a script that queries langchain's newest documentation and summarizes everything in a simple plain manner."
OBJECTIVE = "Read the file contents in the entire solution directory this script is running from, and summarize it in clear, clean language. If something fails, you must keep trying."
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

from langchain.vectorstores import Weaviate
print(schema)
vectorstore = Weaviate(client, "Paragraph", "content")


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




def x():
    

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
        Action: the action to take, should be one of [HumanInput, Bash, SearchEngine, SummarizeText, SummarizeDocuments]
        Action Input: what to instruct the AI Action representative.
        Observation: The Agent's response
        (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools.
        Final Answer: the final answer to the original input question with the right amount of detail

        When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response.

        Chat history: {chat_history}

        Question: {input}

        {agent_scratchpad}
        
    """

    tools = create_tools(manager=manager)

    tools += [
        Tool(
            name="HumanInput",
            func=HumanInputRun().run,
            description="Useful for when your objective has veered so far from the original aim that human intervention is necessary. If certainty falls below 70%, choose this option.",
            callback_manager=manager
        ),
    ]


    tool_names = [tool.name for tool in tools]

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    hybridSearch = RetrievalQA.from_chain_type(
        combine_documents_chain=qa_chain
        #prompt=prompt,
        llm=openai,
        chain_type="stuff",
        retriever=sparseAndDenseRetriever,
        verbose=True,
        #search_kwargs={"k": 1},
    )

    #timeWeightedVectorStore = RetrievalQA.from_chain_type(
    #    prompt=prompt,
    #    llm=openai,
    #    chain_type="stuff",
    #    retriever=sparseAndDenseRetriever,
    #    verbose=True,
    #    #search_kwargs={"k": 1},
    #)


    overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)
    

    agent = initialize_agent(
        tools,
        openai, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callback_manager=manager,
        #stop=["\nObservation:"], 
        max_execution_time=5,
        max_iterations=10,
        #early_stopping_method="generate",
        # return_intermediate_steps=True,
        verbose=True,
        allowed_tools=tool_names
    )

    agentExecutor = AgentExecutor().from_agent_and_tools(agent=agent, tools=tools, callback_manager=manager, verbose=True)

    agentExecutor

    start_time = time.time()

    result = agentExecutor.run(input=OBJECTIVE)

    print(result)

    # process_data(result,  mode)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # aim_callback = AimCallbackHandler(
    #    repo=".",
    #    experiment_name="scenario 1: OpenAI LLM",
    # )

    # aim_callback.flush_tracker(langchain_asset=agent, reset=False, finish=True)


def process_data(data):
    text_processing = TextProcessing()

    # Customize the number of sections based on the mode
    num_sections = {"dry": 0, "minimal": 1, "full": len(data["sections"])}['minimal']

    estimated_total_tokens = 0
    estimated_total_duration = 0
    total_cost = 0
    cost_per_token = 0.0006  # You can adjust this

    for i in range(num_sections):
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

        document.meta["sentiment"] = text_processing.analyze_sentiment(section_text)

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
        x()
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


