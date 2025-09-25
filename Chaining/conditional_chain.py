from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal
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



class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')
    #--> Using Literal because only positive / negative feedback should be shown. No other input is needed. 
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)
#--> using pydantic output parser to add strict constrains to the LLM.


# prompt 1 : used for getting the sentiment of the feedback
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}" ,
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# prompt 2 : used if sentiment is positive
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

# prompt 3 : used if sentiment is negative
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)


#chains
classifier_chain   = prompt1 | model | parser2
positive_chain     = prompt2 | model | parser
negative_chain     = prompt3 | model | parser


# --> branch chain. using RunnableBranch
brach_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', positive_chain),
    (lambda x : x.sentiment == 'negative', negative_chain),
    RunnableLambda( lambda x : "could not find sentiment")
)

final_chain = classifier_chain | brach_chain
print(final_chain.invoke({'feedback':'This is a horrible phone'}))
print('----------------------------------------------Graph-------------------------------')

print(final_chain.get_graph().print_ascii())