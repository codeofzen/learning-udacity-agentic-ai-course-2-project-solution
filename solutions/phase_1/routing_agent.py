import os
from dotenv import load_dotenv
from workflow_agents.base_agents import (
    RoutingAgent,
    KnowledgeAugmentedPromptAgent,
)

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

# Define the Texas Knowledge Augmented Prompt Agent
knowledge = "You know everything about Texas"

texas_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key, persona=persona, knowledge=knowledge
)

# Define the Europe Knowledge Augmented Prompt Agent
knowledge = "You know everything about Europe"
europe_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key, persona=persona, knowledge=knowledge
)

# Define the Math Knowledge Augmented Prompt Agent
persona = "You are a college math professor"
knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
math_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key, persona=persona, knowledge=knowledge
)

routing_agent = RoutingAgent(openai_api_key, {})
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_knowledge_agent.respond(x),
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_knowledge_agent.respond(x),
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_knowledge_agent.respond(x),
    },
]

routing_agent.agents = agents

# Print the RoutingAgent responses to the following prompts:
#           - "Tell me about the history of Rome, Texas"
#           - "Tell me about the history of Rome, Italy"
#           - "One story takes 2 days, and there are 20 stories"
input_texas = "Tell me about the history of Rome, Texas"
print("====================================")
print("Input: ", input_texas)
print(f"Answer: {routing_agent.route_to_agent(input_texas)}")
print("====================================\n")

print("====================================")
input_europe = "Tell me about the history of Rome, Italy"
print("Input: ", input_europe)
print(f"Answer: {routing_agent.route_to_agent(input_europe)}")
print("====================================\n")

print("====================================")
input_math = "One story takes 2 days, and there are 20 stories"
print("Input: ", input_math)
print(f"Answer: {routing_agent.route_to_agent(input_math)}")
print("====================================\n")
