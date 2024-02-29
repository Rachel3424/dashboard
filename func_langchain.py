from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.documents import Document
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
import configparser
from datetime import datetime
from pinecone import Pinecone
import tiktoken
import warnings
warnings.filterwarnings('ignore')

# Setup
config = configparser.ConfigParser()
config.read('config.ini')

class Rachel_langchain():
    TOKEN_PRICE = 0.00010/1000
    encoding = tiktoken.get_encoding("cl100k_base")

    openai = OpenAI(api_key=config.get('OpenAI','api_key'), organization=config.get('OpenAI','organization'), temperature=0)
    embeddings_model = OpenAIEmbeddings(openai_api_key=config.get('OpenAI','api_key'))
    similarity_matches = []

    def __init__(self, index_name, namespace_name, history):
        self.index_name = index_name
        self.namespace_name = namespace_name
        self.detailed_history = {}
        self.total_cost = 0
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
        if not history=='':
            self.intial_history(history)

    def intial_history(self, history):
        seperated_history = [i.split(': ')[1] for i in history.split('\n')]
        #print(f'seperated_history: {seperated_history}')
        print(f'seperated_history***************')
        print(seperated_history)
        for index in range(0, len(seperated_history), 2):
            self.memory.save_context({'question':seperated_history[index]}, {'output':seperated_history[index+1]})
        print('just after for LOO{*****&&&&&&*****}:')
        print(self.memory.load_memory_variables({})['chat_history'])

    def decide_db_or_llm(self, question: str):
        def init_pinecone(index_name):
            pc = Pinecone(
                api_key = config.get('pinecone','api_key'),
                environment='gcp-starter'
            )
            index = pc.Index(index_name)
            return index

        def get_embedding(question):
            # Get embedding
            embedded_quesiton = self.embeddings_model.embed_query(question)

            # Get token + price
            num_tokens = len(self.encoding.encode(question))

            # Store detailed history
            self.detailed_history['embedding_token'] = num_tokens 
            self.detailed_history['embedding_price'] = num_tokens*self.TOKEN_PRICE
            self.total_cost += self.detailed_history['embedding_price']

            return embedded_quesiton
            
        def search_from_pinecone(index, query_embedding, k):
            return index.query(vector=query_embedding, top_k=k, include_metadata=True, namespace=self.namespace_name)

        # Record start time
        self.detailed_history['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Record question
        self.detailed_history['question'] = question

        # Decide db or llm
        #print('Initial pincone...')
        index = init_pinecone(self.index_name)
        #print('Get question embedding...')
        query_embedding = get_embedding(question)
        #print('Search similarity from pinecone...')
        qa_results = search_from_pinecone(index, query_embedding, k=1)
        
        self.similarity_matches=[]
        for every_info in qa_results['matches']:
            # If score >= 0.81
            if every_info['score'] >= 0.95:
                temp={}
                temp['question']=every_info['metadata']['question']
                temp['answer']=every_info['metadata']['answer']
                temp['score']=every_info['score']

                # print(f"question: {every_info['metadata']['question']}")
                # print(f"score: {every_info['score']}")

                self.similarity_matches.append(temp)
        
        if self.similarity_matches==[]:
            return 'ai'
        else:
            return 'db'
        
    def process_by_db(self, question: str, answer_by_llm = False):
        if answer_by_llm:
            # Define prompt
            prompt_template = """
            寫下以下內容的簡明摘要:
            "{text}"
            簡明扼要摘要:"""
            prompt = PromptTemplate.from_template(prompt_template)

            # Define LLM chain
            llm_chain = LLMChain(llm=self.openai, prompt=prompt)

            # Define StuffDocumentsChain
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            all_text_from_similarity = ' '.join([i['answer'] for i in self.similarity_matches])
            documents = [Document(page_content=all_text_from_similarity)]

            #print('- Split text (chunk_size = 500, chunk_overlap = 100)...')
            # Split text to docs
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
            docs = text_splitter.split_documents(documents)

            #print(stuff_chain.run(docs))
            with get_openai_callback() as cb:
                response = stuff_chain.run(docs)

                # Store detailed history
                QA_token, QA_prompt_token, QA_completion_token, _, QA_price = [i.strip() for i in str(cb).split('\n')]
                self.detailed_history['QA_token'] = int(QA_token[13:])
                self.detailed_history['QA_prompt_token'] = int(QA_prompt_token[15:])
                self.detailed_history['QA_completion_token'] = int(QA_completion_token[19:])
                self.detailed_history['QA_price'] = QA_price[19:]

                self.detailed_history['answer'] = response.strip()
                self.detailed_history['answer_type'] = 'db + llm_processed' 

                self.total_cost += float(self.detailed_history['QA_price'])
                self.detailed_history['total_price'] = self.total_cost

            # Save to memory
            self.memory.save_context({'question':question}, {'output':response.replace('\n','')+'。'})

            return response+'。'

        # template = """
        # {chat_history}

        
        # 內容: Summarize this content: {context}
        # 問題: {question}

        # 專業的答案:"""

        # PROMPT = PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])

        
        #     # Get qa chain input
        #     #print('- Get langchain correct input type...')
        #     all_text_from_similarity = ' '.join([i['answer'] for i in self.similarity_matches])
        #     documents = [Document(page_content=all_text_from_similarity)]

        #     #print('- Split text (chunk_size = 500, chunk_overlap = 100)...')
        #     # Split text to docs
        #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        #     docs = text_splitter.split_documents(documents)

        #     #print('- Get response from similarity topK (chain_type = refine)...')
        #     # Get response from qa_chain
        #     with get_openai_callback() as cb:
        #         chain = load_qa_chain(llm=self.openai, chain_type="stuff", prompt=PROMPT, memory=self.memory)
        #         response = chain.run(input_documents=docs, question=question)

        #         # Store detailed history
        #         QA_token, QA_prompt_token, QA_completion_token, _, QA_price = [i.strip() for i in str(cb).split('\n')]
        #         self.detailed_history['QA_token'] = int(QA_token[13:])
        #         self.detailed_history['QA_prompt_token'] = int(QA_prompt_token[15:])
        #         self.detailed_history['QA_completion_token'] = int(QA_completion_token[19:])
        #         self.detailed_history['QA_price'] = QA_price[19:]

        #         self.detailed_history['answer'] = response.strip()
        #         self.detailed_history['answer_type'] = 'db + llm_processed' 

        #         self.total_cost += float(self.detailed_history['QA_price'])
        #         self.detailed_history['total_price'] = self.total_cost

        #     return response
        else:
            db_answer = self.similarity_matches[0]['answer']

            # Store question, answer to memory
            # self.memory.save_context({"question": question}, {"output": db_answer})

            # Store detailed history
            self.detailed_history['QA_token'] = 0
            self.detailed_history['QA_prompt_token'] = 0
            self.detailed_history['QA_completion_token'] = 0
            self.detailed_history['QA_price'] = '$0'

            self.detailed_history['answer'] = db_answer
            self.detailed_history['answer_type'] = "database"
            self.detailed_history['total_price'] = self.total_cost

            # Save to memory
            self.memory.save_context({'question':question}, {'output':db_answer})
            
            return db_answer
    
    def process_by_llm(self, question: str):
        print('before history:')
        print(self.memory.load_memory_variables({})['chat_history'])
        template = """
        你是專業的法律顧問機器人，請用繁體中文回答以下問題，並回答30個字左右

        {chat_history}

        問題: {question}

        專業的答案:"""

        PROMPT = PromptTemplate(template=template, input_variables=["chat_history", "question"])
        
        with get_openai_callback() as cb:
            llm_chain = LLMChain(llm=self.openai, prompt=PROMPT, memory=self.memory)
            response = llm_chain.run(question)

            # Store detailed history
            QA_token, QA_prompt_token, QA_completion_token, _, QA_price = [i.strip() for i in str(cb).split('\n')]
            self.detailed_history['QA_token'] = int(QA_token[13:])
            self.detailed_history['QA_prompt_token'] = int(QA_prompt_token[15:])
            self.detailed_history['QA_completion_token'] = int(QA_completion_token[19:])
            self.detailed_history['QA_price'] = QA_price[19:]

            self.detailed_history['answer'] = response.strip()
            self.detailed_history['answer_type'] = 'llm' 

            self.total_cost += float(self.detailed_history['QA_price'])
            self.detailed_history['total_price'] = self.total_cost

        return response

    def answer_question(self, question: str):
        # Decide who answer the question
        if self.decide_db_or_llm(question) == 'db':
            response = self.process_by_db(question, True)
        else:
            response = self.process_by_llm(question)

        return response, self.detailed_history

    def __repr__(self):
        return self.memory.load_memory_variables({})['chat_history']
    
    def get_history(self):
        return self.memory.load_memory_variables({})['chat_history']
