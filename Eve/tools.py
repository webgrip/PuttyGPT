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
            description="Useful for when your objective has veered so far from the original aim that human intervention is necessary. If certainty falls below 70%, choose this option.",
            callback_manager=manager
        ),
        Tool(
            name="bash",
            func=BashProcess(return_err_output=True).run,
            description="Useful for when you need to do anything on your file system, read and write to files, and in general do everything possible when you can look up code and write the best version of it. Tip: you can write code with this, and even deploy new versions of yourself. Refactor yourself.",
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
            name="Search online for the newest information on current events and human discourse about topics.",
            func=SearxSearchWrapper(searx_host=SEARXNG_URL).run,
            description="Search online for the newest information on current events and human discourse about topics. Only do this if you exhaust all other options. We want to stay low resource intensive.",
            callback_manager=manager
        ),
        Tool(
            name="Summarize a piece of text",
            func=load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce").run,
            description="Useful for when you need to summarize a small or even large piece of text, but not a set of documents. Give a well thought through, intelligent reasonable summarization. The input to this tool should be a string, which is the text that needs to be summarized",
            callback_manager=manager
        ),
        Tool(
            name="Summarize multiple documents",
            func=load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce").run,
            description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
            callback_manager=manager
        )
    ]
    
    return tools