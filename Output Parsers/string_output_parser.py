from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_token = os.getenv('HUGGINGFACE_API_TOKEN')
login(api_token)


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-Prover-V2-671B",
    task= "text-generation"
)

model = ChatHuggingFace(llm=llm)


parser = StrOutputParser()  # --> Automatic fetch the 'result.content' part from the output given by the LLM. No need to input it separately. 



# ---> 1st Prompt : Detailed Report 
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# --> 2nd Prompt: Summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n{text}",
    input_variables=['text']
)


# --> using user input 
def chain_parser():
    user_prompt = input(str('Enter your prompt : '))
    chain = template1 | model | parser | template2 | model | parser
    result = chain.invoke(user_prompt)
    print(result)
    


chain_parser()