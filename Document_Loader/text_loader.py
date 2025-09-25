# Document Loader 1 --> TextLoader 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()
api_token = os.getenv('HUGGINGFACE_API_TOKEN')
login(api_token)


# setting up the model 
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-Prover-V2-671B",
    task= "text-generation"
)
model = ChatHuggingFace(llm=llm)


#--> prompt
prompt = PromptTemplate(
    template= 'Write a summary of the given text \n {text}',
    input_variables=['text']
)


#--> parser 
parser = StrOutputParser()


# --> Document loader
loader = TextLoader(
    'earth.txt',
    encoding='utf-8'
)
doc = loader.load()


# chain 
chain = prompt | model | parser 
result = chain.invoke({'text' : doc[0].page_content})
print(result)