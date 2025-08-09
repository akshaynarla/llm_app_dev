from langchain.chat_models import init_chat_model
import dotenv

# Load API key from .env file
# load_dotenv()
key_location = dotenv.dotenv_values("./geneai.env")
google_api_key = key_location['GOOGLE_API_KEY']

# Configure the Generative AI model. 
# In LangChain,this is done using the `init_chat_model` function.
# Here we are using the Gemini 2.5 Flash model from Google GenAI. You can invoke different models by writing their names.
flash = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=google_api_key)

# invokes the model with a prompot and returns a response object.
# The response object contains the content as well as metadata like token usage, etc.
response = flash.invoke("Explain AI in 25 words.")
print(response.content)  # Print only the content of the response. 