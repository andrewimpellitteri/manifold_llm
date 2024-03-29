from manifoldpy import api
from manifoldpy.api import BinaryMarket
from llama_cpp import Llama
from chatformat import format_chat_prompt
import numpy as np
import os
import pickle

from question_research import research


test_model_path = "/Users/andrew/Documents/dev/text-generation-webui/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
question_list_fname = "question_list.pkl"


class Question:
    def __init__(self, qid, question, market_probability):
        self.qid = qid
        self.question = question
        self.market_probability = market_probability
        self.models = {}
        self.context = []

    def add_model(self, model_name, model_prob):
        self.models[model_name] = model_prob

    def __str__(self):
        models = "\n".join([f"{key}: {value}" for key, value in self.models.items()])
        return f"Question ID: {self.qid}\nQuestion: {self.question}\nMarket Probability: {self.market_probability}\n{models}"


# Save question_list to a file using pickle
def save_question_list(question_list, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(question_list, file)


# Load question_list from a file using pickle
def load_question_list(file_path):
    if not os.path.exists(file_path):
        return []

    with open(file_path, "rb") as file:
        return pickle.load(file)


def logprob_convert(logprob):
    return 1 / (1 + np.exp(-logprob))


def process_question(question, llm, prompt_format="llama-2"):
    
    
    # sys_prompt = "You are a helpful assistant who is trying to help me bet on a prediction market and always answers with a single word, either yes or no. You will only respond to the question with a single word, either yes or no. You will be penalized if you do not respond with yes or no. If you are unsure you must respond with the option you think has the higher probability."
    alt_sys = "You are a helpful assistant specializing in predicting outcomes. Always respond with a single word, either 'yes' or 'no.' You will be penalized for deviating. If unsure, choose the option with the higher probability based on recent information."

    prompt, stop = format_chat_prompt(
        template=prompt_format,
        messages=[
            {"role": "system", "content": alt_sys},
            {"role": "user", "content": question},
        ],
    )

    output = llm(prompt, max_tokens=32, stop=stop, logprobs=True, temperature=0)
    output_text = output["choices"][0]["text"]

    print("output text ", output_text)

    prob = None
    if "yes" in output_text.lower() or "no" in output_text.lower():
        token_probs = output["choices"][0]["logprobs"]["top_logprobs"]

        # Iterate through the keys of the dictionary
        for entry in token_probs:
            key, result = entry.popitem()
            if "yes" in key.lower():
                # Get the result for the key

                prob = logprob_convert(result)

            else:
                prob = 1 - logprob_convert(result)

            return prob


def run_model_on_questions(model_path, question_list, prompt_format="llama-2"):
    llm = Llama(model_path=model_path, use_mlock=True, logits_all=True)

    model_name = os.path.basename(model_path)

    for question in question_list:
        model_prob = process_question(
            question.question, llm, prompt_format=prompt_format
        )

        if model_prob is not None:
            print(question)

            print(f"llm {model_name}: {model_prob}")

            question.models[model_name] = model_prob

            save_question_list(question_list, question_list_fname)


def add_questions(question_list, question_limit=1000):
    for market in api.get_all_markets(limit=question_limit):
        if isinstance(market, BinaryMarket):
            existing_question = next(
                (question for question in question_list if question.qid == market.id),
                None,
            )

            if existing_question is not None:
                # Update market probability if it has changed
                if existing_question.market_probability != market.probability:
                    existing_question.market_probability = market.probability
                    print(
                        f"Updated market probability for Question ID {market.id}: {market.probability}"
                    )

                continue

            question_list.append(
                Question(market.id, market.question, market.probability)
            )

    save_question_list(question_list, file_path=question_list_fname)


def do_research(searches):
    for question in question_list:
        question.context = research(question.question)


question_list = load_question_list(question_list_fname)

add_questions(question_list)

run_model_on_questions(test_model_path, question_list, prompt_format="llama-2")
