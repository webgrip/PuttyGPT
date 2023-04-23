#from agents.AutoGPT import AutoGPT
#from agents.BabyAGI import BabyAGI



from agents.GenerativeAgent import AutonomousAgent
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer


tracer = LangChainTracer()
tracer.load_default_session()
callback_manager = CallbackManager([StdOutCallbackHandler(), tracer])

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





#ryanMemories = [
#    "Ryan remembers his first job, where he was working on a new concept for webshops at a startup",
#    "Ryan feels energetic today",
#    "Ryan knows a lot about writing clean code. Clean architecture.",
#    "Ryan is a world class programmer",
#    "Ryan is completely devoted to the goal of creating AGI, and every step he takes is one on that path",
#]


## The current "Summary" of a character can't be made because the agent hasn't made
## any observations yet.
#print(agent.get_summary())

## We can give the character memories directly

#for memory in ryanMemories:
#    agent.add_memory(memory)
## Now that Tommie has 'memories', their self-summary is more descriptive, though still rudimentary.
## We will see how this summary updates after more observations to create a more rich description.
#print(agent.get_summary(force_refresh=True))