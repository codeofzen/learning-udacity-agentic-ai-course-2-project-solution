# TODO: 1 - Import the AugmentedPromptAgent class
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = (
    "You are a college professor; your answers must always start with: 'Dear students,'"
)

# Instantiate an object of AugmentedPromptAgent with the required parameters
agent = AugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona)

# Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# Add a comment explaining:
# The agent uses the knowledge stored in its training data to answer the prompt.
print("The agent uses the knowledge stored in its training data to answer the prompt.")
# The agent uses a system prompt to adopt a formal tone, starting its response with 'Dear students,'
print(
    "The agent uses a system prompt to adopt a formal tone, starting its response with 'Dear students,'"
)
