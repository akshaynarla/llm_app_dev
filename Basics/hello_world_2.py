from langchain.chat_models import init_chat_model
import dotenv

# Load API key from .env file
# load_dotenv()
key_location = dotenv.dotenv_values("./geneai.env")
google_api_key = key_location['GOOGLE_API_KEY']

# Configure the Generative AI model. 
# In LangChain,this is done using the `init_chat_model` function.
# Here we are using the Gemini 2.5 Flash model from Google GenAI. You can invoke different models by writing their names.
flash = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=google_api_key,
                        configurable_fields=("temperature"))

prompt = "Explain the differences between unit testing and integration testing."

for chunk in flash.with_config({"temperature":0.9}).stream(prompt):
    print(chunk.content, end="")  # Print each chunk of the response content without a newline

