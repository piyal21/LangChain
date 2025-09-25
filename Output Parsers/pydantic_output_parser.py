from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser
from pydantic import BaseModel , Field
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



class Person(BaseModel):
    name : str = Field(description='Name of the person')
    age : int = Field(gt= 18 , description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')
    
    
parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template= 'Generate the name, age and city of the fictional {place} person \ n{format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# using chain 
def llm_result():
    user_prompt = input(str("Enter the name of any country : "))
    chain = template | model | parser
    llm_output = chain.invoke(user_prompt)
    print(llm_output)
    
llm_result()