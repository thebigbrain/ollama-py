import numpy as np

from xlab.generative_agent.similarity import Similarity
from xlab.generative_agent.state import EnvState
from xlab.generative_agent.experience import Experience


class MemoryStream:
    _stream: list[Experience]

    def add(self, experience: Experience):
        self._stream.append(experience)

    def __iter__(self):
        return iter(self._stream)


class MemoryStreamModule:
    def __init__(self):
        self.memory_stream = MemoryStream()

    def add_experience(self, experience: Experience):
        self.memory_stream.add(experience)

    def retrieve_memories(self, state: EnvState):
        return retrieve_memories(
            self.memory_stream,
        )

    def form_long_term_plan(self):
        # Analyze memories and form a long-term plan
        # This is a placeholder for a more complex implementation
        long_term_plan = None
        return long_term_plan

    def reflect(self):
        # Reflect on past experiences and gain insights
        # This is a placeholder for a more complex implementation
        reflections = None
        return reflections


def retrieve_memories(
        memory_stream: MemoryStream,
        current_state :EnvState,
        recency_weight,
        importance_weight,
        relevance_weight,
        step_number,
        similarity: Similarity,
        K=10,
):
    # Calculate relevance score for each memory
    relevance_scores = []
    for memory in memory_stream:
        # Calculate recency component
        recency_score = 1.0 / (
                step_number - memory.reward
        )  # Assuming reward is at index 2

        # Calculate importance component
        importance_score = abs(memory.reward)

        # Calculate relevance component (Replace with your chosen similarity function)
        relevance_score = similarity(
            memory.perception, current_state
        )  # Placeholder, define similarity function

        # Calculate overall relevance score
        overall_score = (
                recency_weight * recency_score
                + importance_weight * importance_score
                + relevance_weight * relevance_score
        )

        relevance_scores.append((memory, overall_score))

    # Sort memories by relevance score in descending order
    sorted_memories = sorted(relevance_scores, key=lambda x: x[1], reverse=True)

    # Select top K memories based on relevance score
    selected_memories = [memory for memory, _ in sorted_memories[:K]]

    return selected_memories
