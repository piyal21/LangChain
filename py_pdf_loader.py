# --> PyPDFLoader loads content from the PDF files and convert each page into a document object.
# a pdf contains 25 pages. ---- pypdfloader will convert it into 25 document objects.
# for simple pdf. not for complex pdf files. 

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import os

load_dotenv()
api_token = os.getenv('HUGGINGFACE_API_TOKEN')
login(api_token)
parser = StrOutputParser()


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


# --> loading the document.
loader = PyPDFLoader('test_file.pdf')
pdf_file = loader.load()

# --> prints the whole document. 
#print(pdf_file[0])

# --> prints the number of pages the pdf holds. 
print(len(pdf_file))

#--> prints the content of the first page. 
print(pdf_file[0].page_content)
