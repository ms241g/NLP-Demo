from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

print('STart the app')
openai.api_key = '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'
#openai.api_key ='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'
app = Flask(__name__)


train=pd.read_csv('QA60.csv')
#train=train[['Unnamed: 0','Unnamed: 1','Unnamed: 2']]
#train=train[(train['Unnamed: 2']=='fundstransfer')]
#train.drop('Unnamed: 2', axis=1,inplace=True)
# train=pd.read_excel('Alternate_Question_Answers.xlsx')
train.columns = ['question','answer']


def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


embed_data = np.load('embeddings60.npy',allow_pickle=True)
train['embedding']=embed_data


# search through the reviews for a specific product
def search_question(train,user_question,n):
    userq_embedding = get_embedding(
        user_question
    )
    train["similarity"] = train.embedding.apply(lambda x: cosine_similarity(x, userq_embedding))

    results = (
        train.sort_values("similarity", ascending=False)
        .head(n)
    )
    qualifying_question = results['question'].tolist()
    qualifying_answer = results['answer'].tolist()
    similarity_score = results['similarity'].tolist()

    return qualifying_question[0], qualifying_answer[0], similarity_score[0]


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(30))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def completion_with_backoff_p(prompt):
    res = completion_with_backoff(model="davinci:ft-personal-2023-01-21-19-19-10",
                                  prompt=prompt,
                                  max_tokens=17,
                                  temperature=0,
                                  top_p=1,
                                  frequency_penalty=0,
                                  presence_penalty=0,
                                  stop=[" END"])

    # return res
    return res['choices'][0]['text']


def summarize_oneshot(prompt):
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0 )

    return response


@app.route('/', methods=["GET", "POST"])
def predict_intent():
    if request.method == 'POST':
        message = dict(request.form)
        if message['Intent Classification'] != '':
            data = message['Intent Classification']
            prompt = data[0] + '\n\n###???\n\n'
            my_prediction = completion_with_backoff_p(prompt)
            my_prediction = "Classified Intent: {intent}".format(intent=my_prediction)
            return render_template('index.html', input=data, prediction=my_prediction)

        elif message['FAQ response'] != '':
            data = message['FAQ response']
            results = search_question(train, data, n=1)
            f_results = "Qualified question: {question} \n\nQualified response: {response} \n\nSimilarity score: {score}".format(question=results[0],response=results[1],score=results[2])
            return render_template('index.html', input=data, prediction= f_results)  #,

        elif message['Summarize'] != '':
            data = message['Summarize']
            print('data', data)
            prompt = 'Summarize- ' + data
            results = summarize_oneshot(prompt)
            results = results['choices'][0]['text']
            results = "Summarized text: {summary}".format(summary=results)
            return render_template('index.html', input=data, prediction=results)

    else:
        return render_template("index.html")


#response = openai.Completion.create(
#  model="text-davinci-003",
#  prompt="Summarize - Hello, I would like to know if there is any way of removing my monthly service fee? Because of the times we are going through I can not afford to pay$ 12 fee. Last statements they were able to remove it, and I would like to\nknow if they can remove it again\n\nThe author is asking if the monthly service fee can be removed due to financial hardship related to the current times. They note that it was removed previously and are asking if it can be done again.",
#  temperature=0.7,
#  max_tokens=256,
#  top_p=1,
#  frequency_penalty=0,
#  presence_penalty=0
#)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)