from open_spiel.python.algorithms import minimax
from gamingbench.agents.base_agent import BaseAgent


class MinimaxAgent(BaseAgent):

    def __init__(self, config, **kwargs):
        super(MinimaxAgent, self).__init__(config)
        self.game = kwargs['game']

    def step(self, observations):
        agent_action_list = observations['legal_moves']
        openspiel_action_list = observations['openspiel_legal_actions']
        state = observations['state']
        action = minimax.alpha_beta_search(self.game, state=state)[1]

        print(agent_action_list)
        print(openspiel_action_list)
        print(action)
        return agent_action_list[openspiel_action_list.index(action)], []


