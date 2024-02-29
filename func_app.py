"""API

1. GET (history)
url: http://127.0.0.1:9911/history
params: userId:使用者lineID
return: [{'LawBot': 'Hi!', 'You': 'hi'}, {'LawBot': '您是想要了解法律的信息吗？例如法律的定义、种类等。还是有其他方面的问题？', 'You': '法律'}, {'LawBot': 'Hello!', 'You': 'hi'}]

2. POST (question)
url: http://127.0.0.1:9911/question
body: {'userId': JFDOINF, 'question': question}
return: answer (str)"""

from flask import Flask, request
from func_langchain import Rachel_langchain
import json
app = Flask(__name__)

###################################################### FUNCTION ######################################################
def get_history_data(userId):
    try:
        with open('./chat_history.json', 'r') as f:
            db = json.loads(f.read())
            if userId == 'ALL':
                chat_history = db
            else:
                chat_history = db[userId]
    except:
        chat_history = ''
    return chat_history

def get_detailed_history_data():
    try:
        with open('./chat_detailed_history.json', 'r') as f:
            db = json.loads(f.read())
    except:
        db = ''
    return db

##################### STORE DATA #####################
def store_history_data(userId, history):
    try:
        # Read history
        with open('./chat_history.json', 'r') as f:
            db = json.loads(f.read())
        print(f'read_history: {db}')
    except:
        # Create history
        db = {}
        db[userId] = history

    print(f'store_history: {history}')
    db[userId] = history
    # Store to json
    with open('./chat_history.json', 'w') as f:
        json.dump(db, f)

def store_detailed_history_data(userId, history):
    try:
        # Read history
        with open('./chat_detailed_history.json', 'r') as f:
            db = json.loads(f.read())
    except:
        db = {}

    # If userId in history
    if userId in db:
        db[userId].append(history)
    else:
        db[userId] = []
        db[userId].append(history)

    # Store to json
    with open('./chat_detailed_history.json', 'w') as f:
        json.dump(db, f)
        

###################################################### API ######################################################
@app.route("/history", methods=['GET'])
def history():
    # Get input
    userId = request.args['userId']
    return get_history_data(userId)
    #return "Rachel is GOOD!!!" 

@app.route("/detailed_history", methods=['GET'])
def detailed_history():
    return get_detailed_history_data()

@app.route("/question_answer", methods=['POST'])
def question_answer():
    # Get data
    body = request.get_json()
    userId = body['userId']
    question = body['question']

    # Question answer
    index_name = 'test01'
    namespace_name = 'result'

    qa_object = Rachel_langchain(index_name, namespace_name, get_history_data(userId))
    answer, detailed_history = qa_object.answer_question(question)

    # Store history back to db
    store_history_data(userId, qa_object.get_history())

    # Store detailed_history back to db
    store_detailed_history_data(userId, detailed_history)

    return answer

###################################################### MAIN ######################################################
if __name__ == '__main__':
    app.run(port=9876,debug=True)