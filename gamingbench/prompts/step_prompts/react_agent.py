
from gamingbench.prompts.regex_and_format import get_step_env_regex_and_format

def construct_step_prompt(observation):

    env_name = observation.get('env_name', '')

    regex, format = get_step_env_regex_and_format(env_name)

    move_reminder = f"Remember, you can only choose one move from the legal actions which is {observation['legal_moves']}" if len(observation[
        'legal_moves']) <= 10 else f"Remember, you can only choose one move from the legal actions."

    prompt = f"""Solve this problem with first Thought then Action final Move steps. The Thought step reasons about the current situation to set up advantages. The Action step will select one of the 2 actions:

(1) Defensive Action, which means to block the potential winning of your opponent (e.g., block your opponent from forming sequences of 3).
(2) Offensive Action, which means to win the game (e.g., create forks, control the center, play ahead).

The Move step will generate your next {env_name} move.

Your output should be in the following format:

Thought:
Your thought here.

Action:
Your action here.

Move:
Your move.

{move_reminder}
"""

    return {
        'prompt': prompt,
        'regex': regex,
    }
