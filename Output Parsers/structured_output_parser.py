# --> we tell the LLM to give output in a specific structured schema like 'JSON' etc. 
# --> The schema is given to LLM by the user. 
# --> Cant do 'Data Validation' in Structured Output Parser. 
# --> to user data validation user can use -- > Pydantic

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

login("hf_bFbaCcuhmObOCNXGAlXGBtbrIpPGcGhLffv")

load_dotenv()

# --> setting up the model cl
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-Prover-V2-671B",
    task= "text-generation"
)
model = ChatHuggingFace(llm=llm)


# making the schema to guide the LLM what type of output we want. 
schema = [
    ResponseSchema(name = 'fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name = 'fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name = 'fact_3',description='Fact 3 about the topic')
]


parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template='Give 3 facts about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# --> NOT USING CHAIN 
# prompt = template.invoke({'topic':'black hole'})
# llm_output = model.invoke(prompt)
# result = parser.parse(llm_output.content)
# print(result)


# --> USING CHAIN
def llm_result():
    user_prompt = input(str('Enter your prompt : '))
    chain = template | model | parser
    llm_output = chain.invoke(user_prompt)
    print(llm_output)


llm_result()



