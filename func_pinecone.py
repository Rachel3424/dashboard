from pinecone import Pinecone
from openai import OpenAI
import configparser
import pandas as pd

# Setup
config = configparser.ConfigParser()
config.read('config.ini')

client = OpenAI(api_key=config.get('OpenAI','api_key'))

# 調用embedding的OpenAPI
def get_embedding(question):
    response = client.embeddings.create(
    model="text-embedding-ada-002",
    input = question
    )
    # 提取生成文本中的嵌入向量
    embedding = response.data[0].embedding
    
    return embedding

# 設定 Pinecone 的初始化參數，創建一個索引對象
def init_pinecone(index_name):
    pc = Pinecone(
        api_key = config.get("pinecone", "api_key"),
        environment='gcp-starter'
    )
    index = pc.Index(index_name)
    return index

# 轉換資料結構
def make_dataset(file_name):
 
    df = pd.read_csv(file_name)
    df.to_dict()

    to_be_upsert = []

    for key, value in df.iterrows():
        temp = {}
        temp['id'] = str(key)
        temp['values'] = get_embedding(value['question'])
        temp['metadata'] = {'question': value['question'],'answer': value['answer'].replace('\n',' ')}

        to_be_upsert.append(temp)

    return to_be_upsert

#################### Main function ####################
def upload_to_pinecone(file_name: str, index_name: str, namespace_name: str):
    index = init_pinecone(index_name)
    index.upsert(
        vectors=make_dataset(file_name),
        namespace=namespace_name)
    

def upload_to_pinecone_no_preprocessing(list_of_dict: list, index_name: str, namespace_name: str):
    index = init_pinecone(index_name)
    index.upsert(
        vectors=list_of_dict,
        namespace=namespace_name)
    
