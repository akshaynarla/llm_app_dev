from langchain_google_genai import ChatGoogleGenerativeAI 
import dotenv
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# Load API key from .env file
# load_dotenv()
key_location = dotenv.dotenv_values("./geneai.env")
google_api_key = key_location['GOOGLE_API_KEY']

# Example prompt for few-shot learning
examples = [
    {"input": "Hello! How are you?", "output": "Hallo! Wie geht's dir?"},
    {"input": "What is your name?", "output": "Wie heißt du?"},
]

# also in the website: https://python.langchain.com/docs/tutorials/llm_chain/
# supports multiple message roles in the prompt.
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("assistant", "{output}")
])

# Configure the Generative AI model in Chat Mode.
flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=google_api_key)

# Create a few-shot prompt template from the example and example prompt for those examples
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt= example_prompt,
    examples= examples
)

# Combine the few-shot prompt into the final prompt template
final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        few_shot_prompt,
        ("human", "{input}")
        ])

# Create a chain that uses the final prompt and the Generative AI model
# pipe to chain different actions.
chain = final_prompt | flash 

# Run the chain with an input
response = chain.invoke({"input": "What is the weather like today?"})
print(response.content)