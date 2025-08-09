from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 
import dotenv

# Load API key from .env file
# load_dotenv()
key_location = dotenv.dotenv_values("./geneai.env")  # Ensure the path to your .env file is correct. Here you need to run from the main folder
google_api_key = key_location['GOOGLE_API_KEY']

# Configure the Generative AI model in Chat Mode.
flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=google_api_key)

# Create a prompt template for generating a poem, instead of directly writing messages/prompt to LLM.
# You can change the number of lines and the topic of the poem as needed.
prompt_template = PromptTemplate.from_template(
    "Write a {number_of_lines} line poem about the {topic}."
)

# Template for zero-shot prompting
# this is particularly useful when you want to parse data from elsewhere.
modelPrompt = prompt_template.format(
    number_of_lines=6,  # You can change the number of lines in the poem
    topic="going to work on Mondays"    # You can change the topic of the poem
)

# Invoke the model with the prompt
response = flash.invoke(modelPrompt)
print(response.content)

