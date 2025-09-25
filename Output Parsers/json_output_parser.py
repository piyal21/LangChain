from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
import os

load_dotenv()
api_token = os.getenv('HUGGINGFACE_API_TOKEN')
login(api_token)


# --> setting up the model . 
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-Prover-V2-671B",
    task= "text-generation"
)
model = ChatHuggingFace(llm=llm)



parser = JsonOutputParser()

#--> making template for json format output
# --> giving dynamic prompt 
template = PromptTemplate(
    template="give me 5 facts about the \n{text}.{format_instruction}",
    input_variables=['text'],
    partial_variables={'format_instruction':parser.get_format_instructions()}  
)


def llm_result(user_prompt):
    # using chain 
    chain = template | model | parser
    result= chain.invoke(user_prompt)
    print(result)
    print()
    print(type(result))
  
    

def user_input():
    user_prompt = input(str('Enter your prompt: '))
    llm_result(user_prompt)
    
    
user_input()


# --> the format of the output is shown by the LLM. 
# --> we can not force a output schema. 
# --> to force an output schema eg: json we need to use structured_output