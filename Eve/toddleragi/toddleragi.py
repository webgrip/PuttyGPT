import time
import os
from collections import deque
from execution_agent import ExecutionAgent
from prioritzation_agent import PrioritizationAgent
from task_creation_agent import TaskCreationAgent
from components.IContextStorage import ContextStorage, ContextData, WeaviateOptions

# Constants

OBJECTIVE = "Write me a script that queries langchain's newest documentation and summarizes everything in a simple plain manner."
INITIAL_TASK = "Decide on what solutions would best serve me to reach my objective"

TASK_STORAGE_NAME = os.getenv("TASK_STORAGE_NAME", os.getenv("TABLE_NAME", "tasks"))
CONTEXT_STORAGE_TYPE = os.getenv("CONTEXT_STORAGE_TYPE", "weaviate")


WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "")
WEAVIATE_VECTORIZER = os.getenv("WEAVIATE_VECTORIZER", "")

assert WEAVIATE_HOST, "WEAVIATE_HOST is missing from .env"
assert WEAVIATE_VECTORIZER, "WEAVIATE_VECTORIZER is missing from .env"


context_storage_options = WeaviateOptions(WEAVIATE_HOST, WEAVIATE_VECTORIZER, TASK_STORAGE_NAME)

context_storage = ContextStorage.factory(CONTEXT_STORAGE_TYPE, context_storage_options)

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
        result = ExecutionAgent(context_storage).run(OBJECTIVE, task["task_name"])
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

        # Enrich result and store in a vector database (weaviate in this case)
        enriched_result = {"data": result}
        result_id = f"result_{task['task_id']}"


        data = { "task": task["task_name"], "result": result }
        context = ContextData(result_id, data, enriched_result['data'])
        context_storage.upsert(context, OBJECTIVE)

        for t in task_manager.task_list:
            print(t)

        # Create new tasks and reprioritize task list
        new_tasks = TaskCreationAgent().run(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            [t["task_name"] for t in task_manager.task_list]
        )
        task_manager.create_new_tasks(new_tasks)
        PrioritizationAgent(task_manager).run(task["task_id"], OBJECTIVE)
        time.sleep(1)

if __name__ == "__main__":
    main()