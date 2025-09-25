from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_token = os.getenv('HUGGINGFACE_API_TOKEN')
login(api_token)

prompt = PromptTemplate(
    template= 'Write 5 interesting facts about {topic}',
    input_variables=['topic']
)

# --> setting up the model. 
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-Prover-V2-671B",
    task= "text-generation"
)
model = ChatHuggingFace(llm=llm)

# --> setting up parser 
parser = StrOutputParser()


# --> using chain 

chain = prompt | model | parser
result = chain.invoke({'topic':'football'})
print(result)
print('--------------------------------------------------------------------------------')
print('---------------------------------The CHAIN----------------------------------')
chain.get_graph().print_ascii()


