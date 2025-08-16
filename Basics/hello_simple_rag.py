import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate

# Load API key from .env file
# load_dotenv()
key_location = dotenv.dotenv_values("./geneai.env")
google_api_key = key_location['GOOGLE_API_KEY']

wiki_search = "Rafael Nadal"

wiki_loader = WikipediaLoader(query= wiki_search, load_max_docs=1)
loaded_data = wiki_loader.load()[0].page_content

template = ChatPromptTemplate.from_messages(
    [" You are a helpful assistant. "
    "Answer the users question in less than 50 words based only on the following context:"
    "Context: {context}" 
    "Question: {question}"])

messages = template.format_messages(
    question = "How many grandslams did Rafael Nadal win?",
    context = loaded_data
)
# print("Prompt looks like:", messages)

# Configure the Generative AI model. 
flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=google_api_key)
response = flash.invoke(messages)
print("LLM responded:" + response.content)