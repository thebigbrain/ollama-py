from typing import Any
import numpy as np

from xlab.generative_agent.state import EnvState, Perception


class Similarity:
    def __call__(self, perception: Perception, current_state: EnvState) -> Any:
        pass


class LinearSimilarity(Similarity):
    def __call__(self, perception: Perception, current_state: EnvState):
        return np.linalg.norm(current_state - perception)


class CosineSimilarity(Similarity):
    def __call__(self, perception: Perception, current_state: EnvState):
        return np.dot(current_state, perception) / (
            np.linalg.norm(current_state) * np.linalg.norm(perception)
        )


class JacardSimilarity(Similarity):
    def __call__(self, perception: Perception, current_state: EnvState):
        return len(current_state & perception) / len(current_state | perception)
