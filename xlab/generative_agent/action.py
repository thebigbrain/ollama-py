class Action:
    pass


class ActionPolicy:
    def take_action(self, memories, state) -> Action:
        raise NotImplemented("action policy not implemented")

    def update(self, state, action, reward, next_state):
        pass
