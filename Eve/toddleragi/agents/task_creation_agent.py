from .openai_connector import OpenAIConnector
from typing import Dict, List

class TaskCreationAgent:
    def __init__(self):
        self

    def run(self, objective: str, result: Dict, task_description: str, task_list: List[str]):
        prompt = f"""
        You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
        The last completed task has the result: {result}.
        This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}.
        Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
        Return the tasks as an array."""
        response = OpenAIConnector().openai_call(prompt) #<------------------------------------------ DANGER!!!
        new_tasks = response.split("\n") if "\n" in response else [response]
        return [{"task_name": task_name} for task_name in new_tasks]




    #prompt = f"""
    #    You are a task creation AI. Your objective is to create new tasks based on the following:
    #    - Objective: {objective}
    #    - Last task completed: {task}
    #    - Result of the last task: {enriched_result['data']}
    #    - Current task list: {task_list}

    #    Generate a list of new tasks to be added to the current task list. Return the result as a list of task names, like:
    #    - First new task
    #    - Second new task
    #    """