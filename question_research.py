from chatformat import format_chat_prompt
import re
from googlesearch import search
from goose3 import Goose
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def summarize(resp):
    model_name = "google/pegasus-xsum"
    device = "mps"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    batch = tokenizer([resp], truncation=True, padding="longest", return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return tgt_text


def generate_questions(question, llm, prompt_format):

    
    sys_prompt = "I want to construct three google searches, seperated by commas to help you answer the following question."

    prompt, stop = format_chat_prompt(
        template=prompt_format,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
    )

    output = llm(prompt, max_tokens=128, stop=stop, temperature=0)
    output_text = output["choices"][0]["text"]

    quoted_text = re.findall(r'"([^"]*)"', output_text)

    searches = []
    for text in quoted_text:

        searches.append(list(search(text, num_results=1))[0])

    return searches


def research(searches):
    g = Goose()

    # summarize_prompt = "I want to you summarize the following piece of text in 3 sentences."
    summaries = []
    for url in searches:
        article = g.extract(url=url)
        print(article.cleaned_text)

        summaries.append(summarize(article.cleaned_text)[0])

    print(summaries)

    return summaries

