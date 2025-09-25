from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()
api_token = os.getenv('HUGGINGFACE_API_TOKEN')
login(api_token)


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-Prover-V2-671B",
    task= "text-generation"
)

model = ChatHuggingFace(llm=llm)


# ---> 1st Prompt : Detailed Report 

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# --> 2nd Prompt: Summary

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n{text}",
    input_variables=['text']
)


# --> using user input 

def prompting(k):
    # --> Detailed Report Generation
    prompt1= template1.invoke({'topic':k})
    result = model.invoke(prompt1)

    # --> Summary Generation
    prompt2 = template2.invoke({'text':result.content})
    result1 = model.invoke(prompt2)

    print(result1.content)
    
    

def main():
    user_input = input(str('Enter your Prompt : '))
    prompting(user_input)
    
if __name__ == "__main__":
    main()