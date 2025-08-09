from langchain_google_genai import ChatGoogleGenerativeAI # to directly use some models, you need to install relevant langchain packages and invoke them.
import dotenv

# Load API key from .env file
# load_dotenv()
key_location = dotenv.dotenv_values("./geneai.env")
google_api_key = key_location['GOOGLE_API_KEY']

# Configure the Generative AI model. 
# Here we are using the Gemini 2.5 Flash model from Google GenAI. You can invoke different models by writing their names.
# temperature is set to 1.1 to get more creative responses. Higher the temperature, more creative the response.
# Note: The temperature should be between 0.0 and 2.0.
flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=google_api_key, temperature=1.1)

prompt = "Write a 5 line poem about the sea."

response = flash.invoke(prompt)
print(response.content)

