from typing import List

from langchain import LLMChain
from langchain.agents import Tool
from langchain.callbacks.base import CallbackManager
from langchain.tools.human.tool import HumanInputRun
from langchain.memory import ReadOnlySharedMemory
from langchain.utilities import BashProcess
from langchain.llms import OpenAI

from langchain.utilities import SearxSearchWrapper
#from wikipedia_api_wrapper import WikipediaAPIWrapper
#from wolfram_alpha_api_wrapper import WolframAlphaAPIWrapper
from config import SEARXNG_URL

from langchain.chains.summarize import load_summarize_chain

def create_tools(llm_chain: LLMChain, memory: ReadOnlySharedMemory, manager: CallbackManager) -> List[Tool]:
    # zapier = ZapierNLAWrapper() Future

    tools = [
        Tool(
            name="human_input_required",
            func=HumanInputRun().run,
            description="useful for when you think you cannot proceed with your task until outside human intervention has occured, OR, when you simply would like some clarification",
            callback_manager=manager
        ),
        Tool(
            name="bash",
            func=BashProcess().run,
            description="useful for when you need to do anything on your file system, read and write to files, and in general do everything possible with access to a command line",
            callback_manager=manager
        ),
        #Tool  (
        #    name="Wolfram",
        #    func=WolframAlphaAPIWrapper().run,
        #    description="useful for when you need to calculate minor to complex math or plot graph data",
        #    callback_manager=manager
        #),
        #Tool(
        #    name="Wikipedia",
        #    func=WikipediaAPIWrapper().run,
        #    description="useful for when you need to fact check or get comprehensive information on a subject, concept or task, ranging from the tiniest thing to the biggest humanity's mind have had to offer",
        #    callback_manager=manager
        #),
        Tool(
            name="search",
            func=SearxSearchWrapper(searx_host=SEARXNG_URL).run,
            description="Only do this if you exhaust all other options",
            callback_manager=manager
        ),
        Tool(
            name="summarize",
            func=load_summarize_chain(OpenAI(temperature=0), chain_type="refine").run,
            description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
            callback_manager=manager
        )
    ]
    
    return tools