
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_loaders import TextLoader

from langchain.document_loaders import TextLoader

import weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.schema import Document
import os
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser, RetryOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ChatMessageHistory




from langchain.chains import LLMChain
from langchain.chains.base import Chain

from typing import Dict, List


from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper



from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper

import os


import pprint
from langchain.utilities import SearxSearchWrapper

search = SearxSearchWrapper(searx_host="http://127.0.0.1:8080", k=5) # k is for max number of items

results = search.results("large language model", num_results = 20, categories='it', time_range='year', engines=['github', 'gitlab', 'arxiv'])
pprint.pp(list(filter(lambda r: r['engines'][0] == 'github', results)))
#'Paris is the capital of France, the largest country of Europe with 550 000 km2 (65 millions inhabitants). Paris has 2.234 million inhabitants end 2011. She is the core of Ile de France region (12 million people).'


### Zapier
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents import AgentType
from langchain.utilities.zapier import ZapierNLAWrapper

## step 0. expose gmail 'find email' and slack 'send channel message' actions

# first go here, log in, expose (enable) the two actions: https://nla.zapier.com/demo/start -- for this example, can leave all fields "Have AI guess"
# in an oauth scenario, you'd get your own <provider> id (instead of 'demo') which you route your users through first

llm = OpenAI(temperature=0)
zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("Summarize the last email I received regarding Silicon Valley Bank. Send the summary to the #test-zapier channel in slack.")


# > Entering new AgentExecutor chain...
#  I need to find the email and summarize it.
# Action: Gmail: Find Email
# Action Input: Find the latest email from Silicon Valley Bank
# Observation: {"from__name": "Silicon Valley Bridge Bank, N.A.", "from__email": "sreply@svb.com", "body_plain": "Dear Clients, After chaotic, tumultuous & stressful days, we have clarity on path for SVB, FDIC is fully insuring all deposits & have an ask for clients & partners as we rebuild. Tim Mayopoulos <https://eml.svb.com/NjEwLUtBSy0yNjYAAAGKgoxUeBCLAyF_NxON97X4rKEaNBLG", "reply_to__email": "sreply@svb.com", "subject": "Meet the new CEO Tim Mayopoulos", "date": "Tue, 14 Mar 2023 23:42:29 -0500 (CDT)", "message_url": "https://mail.google.com/mail/u/0/#inbox/186e393b13cfdf0a", "attachment_count": "0", "to__emails": "ankush@langchain.dev", "message_id": "186e393b13cfdf0a", "labels": "IMPORTANT, CATEGORY_UPDATES, INBOX"}
# Thought: I need to summarize the email and send it to the #test-zapier channel in Slack.
# Action: Slack: Send Channel Message
# Action Input: Send a slack message to the #test-zapier channel with the text "Silicon Valley Bank has announced that Tim Mayopoulos is the new CEO. FDIC is fully insuring all deposits and they have an ask for clients and partners as they rebuild."
# Observation: {"message__text": "Silicon Valley Bank has announced that Tim Mayopoulos is the new CEO. FDIC is fully insuring all deposits and they have an ask for clients and partners as they rebuild.", "message__permalink": "https://langchain.slack.com/archives/C04TSGU0RA7/p1678859932375259", "channel": "C04TSGU0RA7", "message__bot_profile__name": "Zapier", "message__team": "T04F8K3FZB5", "message__bot_id": "B04TRV4R74K", "message__bot_profile__deleted": "false", "message__bot_profile__app_id": "A024R9PQM", "ts_time": "2023-03-15T05:58:52Z", "message__bot_profile__icons__image_36": "https://avatars.slack-edge.com/2022-08-02/3888649620612_f864dc1bb794cf7d82b0_36.png", "message__blocks[]block_id": "kdZZ", "message__blocks[]elements[]type": "['rich_text_section']"}
# Thought: I now know the final answer.
# Final Answer: I have sent a summary of the last email from Silicon Valley Bank to the #test-zapier channel in Slack.
#
# > Finished chain.


### More explicit control

from langchain.llms import OpenAI
from langchain.chains import LLMChain, TransformChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.tools.zapier.tool import ZapierNLARunAction
from langchain.utilities.zapier import ZapierNLAWrapper

## step 0. expose gmail 'find email' and slack 'send direct message' actions

# first go here, log in, expose (enable) the two actions: https://nla.zapier.com/demo/start -- for this example, can leave all fields "Have AI guess"
# in an oauth scenario, you'd get your own <provider> id (instead of 'demo') which you route your users through first

actions = ZapierNLAWrapper().list()

## step 1. gmail find email

GMAIL_SEARCH_INSTRUCTIONS = "Grab the latest email from Silicon Valley Bank"

def nla_gmail(inputs):
    action = next((a for a in actions if a["description"].startswith("Gmail: Find Email")), None)
    return {"email_data": ZapierNLARunAction(action_id=action["id"], zapier_description=action["description"], params_schema=action["params"]).run(inputs["instructions"])}
gmail_chain = TransformChain(input_variables=["instructions"], output_variables=["email_data"], transform=nla_gmail)

## step 2. generate draft reply

template = """You are an assisstant who drafts replies to an incoming email. Output draft reply in plain text (not JSON).

Incoming email:
{email_data}

Draft email reply:"""

prompt_template = PromptTemplate(input_variables=["email_data"], template=template)
reply_chain = LLMChain(llm=OpenAI(temperature=.7), prompt=prompt_template)

## step 3. send draft reply via a slack direct message

SLACK_HANDLE = "@Ankush Gola"

def nla_slack(inputs):
    action = next((a for a in actions if a["description"].startswith("Slack: Send Direct Message")), None)
    instructions = f'Send this to {SLACK_HANDLE} in Slack: {inputs["draft_reply"]}'
    return {"slack_data": ZapierNLARunAction(action_id=action["id"], zapier_description=action["description"], params_schema=action["params"]).run(instructions)}
slack_chain = TransformChain(input_variables=["draft_reply"], output_variables=["slack_data"], transform=nla_slack)

## finally, execute

overall_chain = SimpleSequentialChain(chains=[gmail_chain, reply_chain, slack_chain], verbose=True)
overall_chain.run(GMAIL_SEARCH_INSTRUCTIONS)


### Wolfram alpha

import os
os.environ["WOLFRAM_ALPHA_APPID"] = "" #$ TODO
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
wolfram = WolframAlphaAPIWrapper()
wolfram.run("What is 2x+5 = -3x + 7?")
# 'x = 2/5'


#### WIKiPEDIA

from langchain.utilities import WikipediaAPIWrapper
wikipedia = WikipediaAPIWrapper()
wikipedia.run('HUNTER X HUNTER')



### SearchNG

# SearchNG SearxNG supports up to 139 search engines.
import pprint
from langchain.utilities import SearxSearchWrapper
search = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")
search.run("What is the capital of France")
# 'Paris is the capital of France, the largest country of Europe with 550 000 km2 (65 millions inhabitants). Paris has 2.234 million inhabitants end 2011. She is the core of Ile de France region (12 million people).'



# Additional params
results = search.results("Large Language Model prompt", num_results=5, categories='science', time_range='year', language='es', engines=['wiki'])
pprint.pp(results)

# github/gitlab
results = search.results("large language model", num_results = 20, engines=['github', 'gitlab'])
pprint.pp(results)

# scientific articles
results = search.results("Large Language Model prompt", num_results=5, engines=['arxiv'])
pprint.pp(results)

# Requests to the outside world

from langchain.utilities import TextRequestsWrapper
requests = TextRequestsWrapper()
requests.get("https://www.google.com")
# Comlete html output of the response

### Using python repl

from langchain.utilities import PythonREPL
python_repl = PythonREPL()
python_repl.run("print(1+1)")
# '2\n'


#### Weather

os.environ["OPENWEATHERMAP_API_KEY"] = ""
from langchain.utilities import OpenWeatherMapAPIWrapper
weather = OpenWeatherMapAPIWrapper()
weather_data = weather.run("London,GB")
print(weather_data)

#```
#In London,GB, the current weather is as follows:
#Detailed status: overcast clouds
#Wind speed: 4.63 m/s, direction: 150°
#Humidity: 67%
#Temperature: 
#  - Current: 5.35°C
#  - High: 6.26°C
#  - Low: 3.49°C
#  - Feels like: 1.95°C
#Rain: {}
#Heat index: None
#Cloud cover: 100%
#```



######## ifttt with spotify

from langchain.tools.ifttt import IFTTTWebhook
import os
key = os.environ["IFTTTKey"]
url = f"https://maker.ifttt.com/trigger/spotify/json/with/key/{key}"
tool = IFTTTWebhook(name="Spotify", description="Add a song to spotify playlist", url=url)
tool.run("taylor swift")



###### Scraping shit off the internet, like documentation or github repos

os.environ["APIFY_API_TOKEN"] = "Your Apify API token"

apify = ApifyWrapper()

loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://python.langchain.com/en/latest/"}]},
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)

index = VectorstoreIndexCreator().from_loaders([loader])

query = "What is LangChain?"
result = index.query_with_sources(query)

print(result["answer"])
print(result["sources"])



#######


# Import things that are needed generically
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
search = SerpAPIWrapper()
human = HumanMessagePromptTemplate
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Music Search",
        func=lambda x: "'All I Want For Christmas Is You' by Mariah Carey.", #Mock Function
        description="A Music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2022?'",
    ),
    Tool(
        name = "Search",
        func=,
        description="useful for when you need to answer questions about current events"
    )
]

agent = initialize_agent(tools, OpenAI(temperature=0), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

####


@tool
def search_api(query: str) -> str:
    """Searches the API for the query."""
    return "Results"

@tool("search", return_direct=True)
def search_api(query: str) -> str:
    """Searches the API for the query."""
    return "Results"

class CustomSearchTool(BaseTool):
    name = "Search"
    description = "useful for when you need to answer questions about current events"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return search.run(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")
    
class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return llm_math_chain.run(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

# Load the tool configs that are needed.
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
tools = [CustomSearchTool(), CustomCalculatorTool()]

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")



#####

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")




##############



class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}


prompt_1 = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = PromptTemplate(
    input_variables=["product"],
    template="What is a good slogan for a company that makes {product}?",
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)
concat_output = concat_chain.run("colorful socks")
print(f"Concatenated output:\n{concat_output}")




################


llm = OpenAI(temperature=0)

conv_memory = ConversationBufferMemory(
    memory_key="chat_history_lines",
    input_key="input",
    return_messages=True
)

summary_memory = ConversationSummaryMemory(llm=OpenAI(), input_key="input")


memory = CombinedMemory(memories=[conv_memory, summary_memory])


conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=memory
)

conversation.predict(input="Hi there!")

dicts = messages_to_dict(conversation.messages)
new_messages = messages_from_dict(dicts)


####




chain = load_chain("lc://chains/llm-math/chain.json")

# you can even save current chains in memory to files: chain.save("file_name.json")



########








#> Entering new ConversationChain chain...
#Prompt after formatting:
#The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

#Current conversation:

#Human: Hi there!
#AI:

#> Finished chain.

#######

template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""
class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")
        
parser = PydanticOutputParser(pydantic_object=Action)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt_value = prompt.format_prompt(query="who is leo di caprios gf?")

bad_response = '{"action": "search"}'

retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=OpenAI(temperature=0))

retry_parser.parse_with_prompt(bad_response, prompt_value)

Action(action='search', action_input='who is leo di caprios gf?')


########


loader = TextLoader('../../../state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

client = weaviate.Client(
    url=os.environ["WEAVIATE_HOST"],
    additional_headers={
        'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]
    }
)

client.schema.delete_all()
client.schema.get()
schema = {
    "classes": [
        {
            "class": "Paragraph",
            "description": "A written paragraph",
            "vectorizer": "text2vec-openai",
              "moduleConfig": {
                "text2vec-openai": {
                  "model": "babbage",
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

######


retriever = WeaviateHybridSearchRetriever(client, index_name="LangChain", text_key="text")

vectorstore = Weaviate(client, "Paragraph", "content")

query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)

print(docs[0].page_content)