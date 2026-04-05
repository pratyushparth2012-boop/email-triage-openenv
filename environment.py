from pydantic import BaseModel
import random

# ---------------- MODELS ---------------- #

class Observation(BaseModel):
    email: str
    step: int
    task: str

class Action(BaseModel):
    action_type: str
    content: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}


# ---------------- ENV ---------------- #

class EmailEnv:

    def __init__(self, task="easy"):
        self.task = task
        self.step_count = 0

        self.dataset = [
            {"text": "Win a free iPhone now!", "label": "spam", "reply": "ignore"},
            {"text": "Meeting at 5 PM", "label": "important", "reply": "ok"},
        ]

    async def reset(self):
        self.current = random.choice(self.dataset)
        self.step_count = 0

        return StepResult(
            observation=Observation(
                email=self.current["text"],
                step=0,
                task=self.task
            ),
            reward=0.0,
            done=False,
            info={}
        )

    async def step(self, action: Action):
        self.step_count += 1
        reward = 0.0

        if self.task == "easy":
            if action.content == self.current["label"]:
                reward = 1.0
            else:
                reward = -1.0

        elif self.task == "medium":
            reward = 0.5 if len(action.content) > 3 else -0.5

        elif self.task == "hard":
            if self.current["reply"] in action.content:
                reward = 1.0
            else:
                reward = -0.5

        done = True

        return StepResult(
            observation=Observation(
                email=self.current["text"],
                step=self.step_count,
                task=self.task
            ),
            reward=reward,
            done=done,
            info={}
        )

    async def close(self):
        pass
