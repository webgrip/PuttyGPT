from babyagi import OpenAIConnector

class PrioritizationAgent:
    def __init__(self, taskmanager):
        self.taskmanager = taskmanager

    def run(self, this_task_id: int):
        task_names = [t["task_name"] for t in self.taskmanager.task_list]
        next_task_id = int(this_task_id) + 1
        prompt = f"""
        You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
        Consider the ultimate objective of your team:{OBJECTIVE}.
        Do not remove any tasks. Return the result as a numbered list, like:
        #. First task
        #. Second task
        Start the task list with number {next_task_id}."""
        response = OpenAIConnector.openai_call(prompt)
        new_tasks = response.split("\n") if "\n" in response else [response]
        self.taskmanager.task_list = deque()
        for task_string in new_tasks:
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                self.taskmanager.addTask({"task_id": task_id, "task_name": task_name})