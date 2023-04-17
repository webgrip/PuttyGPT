import time
import openai
from collections import deque
from execution_agent import ExecutionAgent
from prioritzation_agent import PrioritizationAgent
from task_creation_agent import TaskCreationAgent

# Constants
OPENAI_API_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.7
OBJECTIVE = "Write me a script that queries langchain's newest documentation and summarizes everything in a simple plain manner."
INITIAL_TASK = "Decide on what solutions would best serve me to reach my objective"

class TaskManager:
    def __init__(self):
        self.task_list = deque([])
        self.task_id_counter = 1

    def add_task(self, task: dict):
        self.task_list.append(task)

    def process_next_task(self):
        return self.task_list.popleft()

    def create_new_tasks(self, new_tasks: list):
        for new_task in new_tasks:
            self.task_id_counter += 1
            new_task.update({"task_id": self.task_id_counter})
            self.add_task(new_task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for task in self.task_list:
            print(f"{task['task_id']}: {task['task_name']}")


class OpenAIConnector:
    def __init__(
        self,
        model: str = OPENAI_API_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        max_tokens: int = 100,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_ada_embedding(self, text: str) -> list:
        text = text.replace("\n", " ")
        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
        return response["data"][0]["embedding"]

    def openai_call(self, prompt: str) -> str:
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
            except openai.error.RateLimitError:
                time.sleep(10)
            else:
                break


def main():
    task_manager = TaskManager()

    # Add the first task
    task_manager.add_task({"task_id": 1, "task_name": INITIAL_TASK})

    # Main loop
    while task_manager.task_list:
        # Print the task list
        task_manager.print_task_list()

        # Process the next task
        task = task_manager.process_next_task()
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(f"{task['task_id']}: {task['task_name']}")

        # Execute the task and store the result
        result = ExecutionAgent.run(OBJECTIVE, task["task_name"])
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

        # Enrich result and store in a vector database (weaviate in this case)
        enriched_result = {"data": result}
        result_id = f"result_{task['task_id']}"
        vector = OpenAIConnector.get_ada_embedding(enriched_result["data"])
        index.upsert([(result_id, vector, {"task": task["task_name"], "result": result})],
                     namespace=OBJECTIVE)

        # Create new tasks and reprioritize task list
        new_tasks = TaskCreationAgent(task_manager).run(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            [t["task_name"] for t in TaskManager.task_list],
        )
        TaskManager.create_new_tasks(new_tasks)
        PrioritizationAgent.run(task["task_id"])
        time.sleep(1)

if __name__ == "__main__":
    main()