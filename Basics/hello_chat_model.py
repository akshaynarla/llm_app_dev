from langchain_google_genai import ChatGoogleGenerativeAI 
import dotenv

# Load API key from .env file
# load_dotenv()
key_location = dotenv.dotenv_values("./geneai.env")
google_api_key = key_location['GOOGLE_API_KEY']

# Configure the Generative AI model in Chat Mode.
flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=google_api_key)

messages = [
    ("system", "You are a creative poet."),
    ("human", "Write a 6 line poem about going to work on Mondays.")
    ]

response = flash.invoke(messages)
print(response.content)
