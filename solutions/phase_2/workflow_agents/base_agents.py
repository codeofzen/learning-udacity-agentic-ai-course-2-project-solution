# TODO: 1 - import the OpenAI class from the openai library
import numpy as np
import openai
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from openai import OpenAI
from enum import Enum


class OpenAIModels(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


class AgentType(Enum):
    DIRECT = "direct"
    AUGMENTED = "augmented"
    KNOWLEDGE_AUGMENTED = "knowledge_augmented"
    RAG_KNOWLEDGE = "rag_knowledge"
    EVALUATION = "evaluation"
    ROUTING = "routing"
    ACTION_PLANNING = "action_planning"


# DirectPromptAgent class definition
class DirectPromptAgent:
    """
    An agent that directly responds to user prompts using the OpenAI API.
    This agent does not have any special capabilities or knowledge augmentation.
    It simply generates a response based on the provided prompt.
    """

    def __init__(self, openai_api_key):
        # Initialize the agent
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )
        response = client.chat.completions.create(
            model=OpenAIModels.GPT_3_5_TURBO.value,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """
    An agent that responds to user prompts with a defined persona.
    """

    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        # Create an attribute for the agent's persona
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )

        # Declare a variable 'response' that calls OpenAI's API for a chat completion.
        system_prompt = f"{self.persona}. Forget previous context."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """
    An agent that responds to user prompts using a defined persona and a specific knowledge base.
    This agent generates responses based only on the provided knowledge.
    """

    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        # Create an attribute to store the agent's knowledge.
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )

        system_prompt = """
        You are {persona} knowledge-based assistant. Forget all previous context.
        
        Instructions:
        - Use only the following knowledge to answer, do not use your own knowledge: {knowledge}
        - Answer the prompt based on this knowledge, not your own.
        """.format(
            persona=self.persona, knowledge=self.knowledge
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": input_text,
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        )

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )
        response = client.embeddings.create(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text[start:end],
                    "chunk_size": end - start,
                    "start_char": start,
                    "end_char": end,
                }
            )

            # FIX: ensure that the last chunk exits the loop correctly. The original code
            # would loop forever since the end would never reach the length of the text.
            if end >= len(text):
                break

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(
            f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["text"].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding="utf-8", index=False)
        print(f"[Embedding] Embeddings saved to embeddings-{self.unique_filename}")
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x)))
        df["similarity"] = df["embeddings"].apply(
            lambda emb: self.calculate_similarity(prompt_embedding, emb)
        )

        best_chunk = df.loc[df["similarity"].idxmax(), "text"]

        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context.",
                },
                {
                    "role": "user",
                    "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content


class EvaluationAgent:
    """
    An agent that evaluates responses from a worker agent based on defined criteria.
    It manages interactions between the worker agent and itself to refine responses until they meet the evaluation criteria.
    """

    def __init__(
        self,
        openai_api_key,
        persona,
        evaluation_criteria,
        worker_agent,
        max_interactions,
    ):
        # Initialize the EvaluationAgent with given attributes.
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        """
        Evaluates the initial prompt by interacting with the worker and evaluator agents.
        It generates a response from the worker agent, evaluates it against the criteria,
        and provides feedback for refinement if necessary.

        Parameters:
        initial_prompt (str): The initial prompt to be evaluated.

        Returns:
        dict: A dictionary containing the final response, evaluation, and number of iterations.
        """
        # This method manages interactions between agents to achieve a solution.
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are {self.persona}. Forget previous context.",
                    },
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0,
            )

            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": instruction_prompt},
                    ],
                    temperature=0,
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1,
        }


class RoutingAgent:
    """
    An agent that routes user prompts to the most suitable agent based on the similarity of the prompt to agent descriptions.
    It computes embeddings for both the user input and agent descriptions to determine the best match.
    """

    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        # Define an attribute to hold the agents, call it agents
        self.agents = agents

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.
        """
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )
        response = client.embeddings.create(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        return response.data[0].embedding

    # Define a method to route user prompts to the appropriate agent
    def route_to_agent(self, user_input):
        """
        Routes the user input to the most suitable agent based on similarity scores.
        It computes embeddings for both the user input and agent descriptions to determine the best match.

        Parameters:
        user_input (str): The user input prompt to be routed.

        Returns:
        str: The response from the best matching agent.
        """

        # Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )
            print(similarity)

            # Add logic to select the best agent based on the similarity score between the user prompt and the agent descriptions
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print("\n=================================================== ")
        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        print("===================================================\n ")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    """
    An agent that extracts steps from a user prompt to complete an action.
    It uses a defined knowledge base provided as string input to the constructor to generate a structured list of steps.
    """

    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        """
        Extracts steps from a user prompt using the OpenAI API.
        The steps are returned as a numbered list based on the agent's knowledge.

        Parameters:
        prompt (str): The user prompt from which to extract steps.

        Returns:
        list: A list of steps extracted from the prompt.
        """

        # Instantiate the OpenAI client using the provided API key
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key
        )

        system_prompt = f"""
        You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. 
        
        Instructions:
        - Return the steps as a numbered list. 
        - Only return the steps in your knowledge. 
        - Forget any previous context. 
        
        This is your knowledge: {self.knowledge}
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        response_text = response.choices[0].message.content.strip()

        # Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")
        steps = [
            step.strip() for step in steps if step.strip() and not step.startswith("#")
        ]

        steps = [
            step for step in steps if re.match(r"^\d+\.", step) or step.startswith("*")
        ]

        return steps
