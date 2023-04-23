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

