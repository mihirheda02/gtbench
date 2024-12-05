
from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.prompts.step_prompts.react_agent import construct_step_prompt


class ReActAgent(PromptAgent):

    def __init__(self, config, **kwargs):
        super(ReActAgent, self).__init__(config)

        self.step_prompt_constructor = construct_step_prompt

