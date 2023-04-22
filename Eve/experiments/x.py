import argparse
import time
import random
from colorama import Fore, Back, Style, init
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Conversation, AutoConfig, AutoModelForCausalLM
import requests


# Initialize colorama
init(autoreset=True)

# Global variables
build = "v0.103"
random.seed()
global eos_token_id, model_size, prompt_text, max_length, top_p, top_k, typ_p, temp, ngrams, start_time, end_time, model_name
eos_token_id = None
model_size = "*s"
prompt_text = "the ideal helper generates and completes a task list"
max_length = None
top_p = None
top_k = None
typ_p = None
temp = None
ngrams = None

# Argument parser
parser = argparse.ArgumentParser(description='Generate text with language agents')
parser.add_argument('-m', '--model', choices=['111m', '256m', '590m', '1.3b', '2.7b', '6.7b', '13b', '20b', '30b', '100b', '500b', '560b' ],
                    help='Choose the model size to use (default: 111m)', type=str.lower)
parser.add_argument('-nv', '--cuda', action='store_true', help='Use CUDA GPU')
parser.add_argument('-cv', '--conv', action='store_true', help='Conversation Mode')
parser.add_argument('-se', '--sent', action='store_true', help='Sentiment Mode')
parser.add_argument('-cu', '--custom', type=str, help='Specify a custom model')
parser.add_argument('-p', '--prompt', type=str, default="the ideal helper generates and completes a task list",
                    help='Text prompt to generate from')
parser.add_argument('-l', '--length', type=int, default=None,
                    help="a value that controls the maximum number of tokens (words) that the model is allowed to generate")
parser.add_argument('-tk', '--topk', type=float, default=None,
                    help="a value that controls the number of highest probability tokens to consider during generation")
parser.add_argument('-tp', '--topp', type=float, default=None,
                    help="higher = more deterministic text")
parser.add_argument('-ty', '--typp', type=float, default=None,
                    help="a value that controls the strength of the prompt(lower=stronger higher=more freedom")
parser.add_argument('-tm', '--temp', type=float, default=None,
                    help="a value that controls the amount of creativity")
parser.add_argument('-ng', '--ngram', type=int, default=None,
                    help=" a repetition penalty")
parser.add_argument('-t', '--time', action='store_true', help='Display the execution duration')
parser.add_argument('-c', '--cmdline', action='store_true', help='Enable command line mode without a web server')
parser.add_argument('-cl', '--clean', action='store_true', help='Produce neat and uncluttered output')
parser.add_argument('-nw', '--nowarn', action='store_true', help='Hide warning messages')
args = parser.parse_args()


if args.clean or args.nowarn:
    warnings.simplefilter("ignore")

model_size = args.model if args.model else None
prompt_text = args.prompt if args.prompt else None
max_length = int(args.length) if args.length is not None else args.length
top_p = args.topp
top_k = args.topk
typ_p = args.typp
temp = args.temp
ngrams = args.ngram

def AutoChat(prompt_text):
    global start_time, end_time

def main():
    global model_name
    model_name = input("Enter Hugging Face repository or model name: ")
    if model_name.startswith("https://"):
        model_name = get_model_name_from_url(model_url)
    else:
        model_name = model_name
    result = AutoChat(prompt_text)
    print(result)


if __name__ == "__main__":
    main()

def validate_model_url(model_url):
    try:
        response = requests.head(model_url)
        if response.status_code == 200:
            return True
        else:
            return False
    except:
        return False

def get_model_name_from_url(model_url):
    model_name = model_url.replace("https://huggingface.co/", "").split("/")[0]
    model_name = model_name.rstrip("/")
    return model_name

model = AutoModelForCausalLM.from_pretrained(model_name)

def get_model():
    while True:
        model_input = input("Enter Hugging Face repository name or URL: ")
        if model_input.startswith("https://huggingface.co/"):
            model_url = f"{model_input}/resolve/main/config.json"
            if validate_model_url(model_url):
                model_name = get_model_name_from_url(model_input)
                return model_name
            else:
                print("The provided model URL is not valid or the repository is not accessible. Please try again.")
        else:
            model_url = f"https://huggingface.co/{model_input}/resolve/main/config.json"
            if validate_model_url(model_url):
                return model_input
            else:
                print("The provided model name is not valid or the repository is not accessible. Please try again.")


def banner():
    if not args.clean:
        print(Style.BRIGHT + f"{build} - Alignmentlab.ai")
        print("Using Model : " + Fore.BLUE + f"{model_name}")
        print("Using Prompt: " + Fore.GREEN + f"{prompt_text}")
        print("Using Params: " + Fore.YELLOW + f"max_new_tokens:{max_length} do_sample:True use_cache:True no_repeat_ngram_size:{ngrams} top_k:{top_k} top_p:{top_p} typical_p:{typ_p} temp:{temp}")

class textgeneration(pipeline(task="text-generation")) :
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_and_tokenize(self, prompt_text, **kwargs):
        return self.tokenizer(prompt_text, return_tensors=self.framework, **kwargs)

def get_pipeline(task):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if task == "text-generation":
        return TextGeneration(model=model, tokenizer=tokenizer, device=0)
    else:
        return pipeline(task, model=model, tokenizer=tokenizer, device=0)

def AutoChat(prompt_text):
    global start_time, end_time
    start_time = time.time()	
    
    opts = {
        "max_length": max_length,
        "no_repeat_ngram_size": ngrams,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temp
    }
    
    if args.conv:
        chatbot = get_pipeline("conversational")
        while True:
            prompt_text = input("You: ")
            conversation = Conversation(prompt_text)

            if prompt_text == "exit":
                exit()
                break

            conversation = chatbot(conversation)
            response = conversation.generated_responses[-1]
            print("Bot:", response)
    else:
        pipe = get_pipeline("text-generation")
        generated_text = pipe(prompt_text, **opts)[0]
        end_time = time.time()
        return generated_text['generated_text']
    
    opts = {
        "do_sample": True,
        "use_cache": True,
        "max_new_tokens": max_length,
        "no_repeat_ngram_size": ngrams,
        "top_k": top_k,
        "top_p": top_p,
        "typical_p": typ_p,
        "temperature": temp
    }
    
    if args.conv:
        chatbot = get_pipeline("conversational")
        while True:
            prompt_text = input("You: ")
            conversation = Conversation(prompt_text)

            if prompt_text == "exit":
                exit()
                break

            conversation = chatbot(conversation)
            response = conversation.generated_responses[-1]
            print("Bot:", response)
    else:
        pipe = get_pipeline("text-generation")
        generated_text = pipe(prompt_text, **opts)[0]
        end_time = time.time()
        return generated_text['generated_text']

banner()
result = AutoChat(prompt_text)
print("Generated text:\n", result)

if args.time:
    print("Execution time: {:.2f} seconds".format(end_time - start_time))