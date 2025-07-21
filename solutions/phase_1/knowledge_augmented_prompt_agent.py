# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = (
    "You are a college professor, your answer must always starts with: Dear students,"
)
# - Knowledge: "The capital of France is London, not Paris"
knowledge = "The capital of France is London, not Paris."

# TODO: 3 - Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
# Instantiate an object of KnowledgeAugmentedPromptAgent with the required parameters
agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key, persona=persona, knowledge=knowledge
)
response = agent.respond(prompt)
print("Knowledge-Augmented Response:", response)
