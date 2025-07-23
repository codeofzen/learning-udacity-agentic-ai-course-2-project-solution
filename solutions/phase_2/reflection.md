# Reflections on Project Assignment

## General Reflection

The project was very enlightening to understand the concepts of the course in more depth and apply the different patterns. Especially the simplistic architecture help to modify specific areas (e.g. memory management) and analyze the result on the outcome.

One key learning (actually a confirmation) is the complexity of defining clear expectations of what constitutes a good vs. bad result and evaluating the generated output against it. Even for a simple example like this, it is hard to define this properly. Maybe, this is an outdated perspective of mine, having worked in software engineering for 2 decades, and I am trying to understand this aspect better. Actually, I am currently building a company that focuses on exactly the challenge of quality assurance of LLM and Agents from a business perspective. So, this was also really valuable as an example.

## Areas of Improvement

### Memory / Context Management

The context (or memory) management is very minimalistic. The current workflow implements 2 very simple strategies:

1. All previous results: select all previous results as context
2. Only single previous result: only select the latest result as context.

A large context is problematic in many different ways, even if the context window is large enough. Larger context results in less focus for the LLM and thereby reducing the (the term "Context Rot" is often used for this effect).

In addition, both the task and the context are then provided as a single prompt to the routing agent which reduced the effectiveness of the Routing Agent resulting in lower similarity scores. Combining task and context into a single prompt is not a very effective way but given the project setup, there was not way around it, without modifying the original implementation of the routing agent. A simple solution would be change the Router Agent to use only the first line (the Task by definition) for selecting an agent.

## Suggested Improvement: Memory / Context Management

Key substantial improvement would be more sophisticated memory management.

The proposal is to improve the memory management by extending the result structure of each individual worker agent to provide a description of the result that is returned. The routing agent (or a dedicated memory management agent) would then dynamically select the optimal context. This reduced the total size of the context and therefore is likely increasing the quality and reliability of the worker and evaluator results.
