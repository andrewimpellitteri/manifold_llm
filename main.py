from manifoldpy import api
from manifoldpy.api import BinaryMarket
from llama_cpp import Llama
from chatformat import format_chat_prompt
import numpy as np
import pandas as pd
import os

test_model_path = "/Users/andrew/Documents/dev/text-generation-webui/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
# llm = Llama(model_path=test_model_path, use_mlock=True, logits_all=True)
df_list = []

if os.path.exists("results.csv"):
    df = pd.read_csv("results.csv")
    print(df.columns)


def logprob_convert(logprob):
    return 1 / (1 + np.exp(-logprob))

def process_question(question, llm, prompt_format='llama-2'):
    # sys_prompt = "You are a helpful assistant who is trying to help me bet on a prediction market and always answers with a single word, either yes or no. You will only respond to the question with a single word, either yes or no. You will be penalized if you do not respond with yes or no. If you are unsure you must respond with the option you think has the higher probability."
    alt_sys = "You are a helpful assistant specializing in predicting outcomes. Always respond with a single word, either 'yes' or 'no.' You will be penalized for deviating. If unsure, choose the option with the higher probability based on recent information."

    prompt, stop = format_chat_prompt(
        template="llama-2",
        messages=[
            {"role": "system", "content": alt_sys},
            {"role": "user", "content": question},
        ],
    )

    output = llm(prompt, max_tokens=32, stop=stop, logprobs=True, temperature=0)
    output_text = output["choices"][0]["text"]

    print('output text ', output_text)

    prob = None
    if 'yes' in output_text.lower() or 'no' in output_text.lower():
        token_probs = output["choices"][0]['logprobs']['top_logprobs']

        # Iterate through the keys of the dictionary
        for entry in token_probs:

            key, result = entry.popitem()
            if 'yes' in key.lower():
                # Get the result for the key
                
                prob = logprob_convert(result)

            else:
                prob = 1 - logprob_convert(result)

            return prob


def get_questions(qids):
    return [api.get_market(market_id=qid) for qid in qids]

def add_new_model_to_csv(model_path, df, add_questions=False, question_limit=1000, prompt_format='llama-2'):

    llm = Llama(model_path=model_path, use_mlock=True, logits_all=True)

    if add_questions:

        for market in api.get_all_markets(limit=question_limit):
            if isinstance(market, BinaryMarket):

                if market.id in df['qid']:
                    continue
            
                prob = process_question(market.question, llm, prompt_format='llama-2')

                if prob is not None:
                    print(f"llm prob: {prob}")
                    model_col = os.path.basename(test_model_path)

                    row = [{"qid": market.id, "market": market.probability, model_col: prob}]

                    temp_df = pd.DataFrame(row)

                    pd.concat([df, temp_df])

                    print(df)

                    df.to_csv("results.csv", index=False)


    for question in get_questions(df['qid'].tolist()):

        prob = process_question(question.question, llm, prompt_format=prompt_format)

        if prob is not None:
            print(f"llm: {prob}")

            df.loc[df['qid'] == question.id, os.path.basename(model_path)] = prob
            df.loc[df['qid'] == question.id, 'market'] = question.probability
            print('saving..')
            df.to_csv('results.csv', index=False)


def create_df(limit):
    for market in api.get_all_markets(limit=limit):
        if isinstance(market, BinaryMarket):

            # print(market.id)
            print(f"Question: {market.question}, market prob: {market.probability}")
            # print(market.probability)
            # print(market.createdTime)
            # print(market)
            # print("\n")

            prob = process_question(market.question)

            if prob is not None:
                print(f"llm prob: {prob}")
                model_col = os.path.basename(test_model_path)

                row = {"qid": market.id, "market": market.probability, model_col: prob}
                df_list.append(row)
                
                df = pd.DataFrame(df_list)

                df.to_csv("results.csv", index=False)

                print(df)



new_model_path = "/Users/andrew/Documents/dev/text-generation-webui/models/openchat_3.5.Q4_K_M.gguf"
add_new_model_to_csv(new_model_path, df, add_questions=True)