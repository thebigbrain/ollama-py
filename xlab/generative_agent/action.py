class Action(int):
    pass


class ActionPolicy:
    def take_action(self, memories, state) -> Action:
        raise NotImplemented("action policy not implemented")
