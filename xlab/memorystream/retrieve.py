def retrieve_memories(
    memory_stream,
    current_state,
    recency_weight,
    importance_weight,
    relevance_weight,
    step_number,
    similarity,
    K,
):
    """
    Retrieves relevant memories from the memory stream based on recency, importance, and relevance.

    Args:
        memory_stream: A list of experiences stored in the memory stream.
            Each experience is a tuple of (perception, action, reward).
        current_state: The current state of the environment.
        recency_weight: Weight assigned to the recency of a memory.
        importance_weight: Weight assigned to the importance of a memory (absolute value of reward).
        relevance_weight: Weight assigned to the relevance of a memory (similarity to current state).
        step_number: The current step number within the training process.

    Returns:
        A list of the top K most relevant memories based on the calculated relevance score.
    """

    # Calculate relevance score for each memory
    relevance_scores = []
    for memory in memory_stream:
        perception, action, reward = memory

        # Calculate recency component
        recency_score = 1.0 / (step_number - memory[2])  # Assuming reward is at index 2

        # Calculate importance component
        importance_score = abs(reward)

        # Calculate relevance component (Replace with your chosen similarity function)
        relevance_score = similarity(
            perception, current_state
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
