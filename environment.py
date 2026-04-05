from pydantic import BaseModel
import random

class Observation(BaseModel):
    email: str
    step: int
    task: str

class Action(BaseModel):
    action_type: str  # classify / reply
    content: str

class EmailEnv:
    def __init__(self, task="easy"):
        self.task = task
        self.step_count = 0

        self.dataset = [
            {
                "text": "Win a free iPhone now!",
                "label": "spam",
                "reply": "Ignore this email."
            },
            {
                "text": "Meeting at 5 PM",
                "label": "important",
                "reply": "Okay, I will attend the meeting."
            },
            {
                "text": "Your order has been shipped",
                "label": "important",
                "reply": "Thank you for the update."
            }
        ]

    def reset(self):
        self.current = random.choice(self.dataset)
        self.step_count = 0

        return Observation(
            email=self.current["text"],
            step=0,
            task=self.task
        )

    def step(self, action: Action):
        self.step_count += 1
        reward = 0

        # 🟢 EASY TASK
        if self.task == "easy":
            if action.action_type == "classify":
                if action.content == self.current["label"]:
                    reward = 1.0
                else:
                    reward = -1.0

        # 🟡 MEDIUM TASK
        elif self.task == "medium":
            if action.action_type == "classify":
                if action.content == self.current["label"]:
                    reward += 0.5
                else:
                    reward -= 0.5

            if action.action_type == "reply":
                if len(action.content) > 5:
                    reward += 0.5

        # 🔴 HARD TASK
        elif self.task == "hard":
            if action.action_type == "reply":
                expected = self.current["reply"].lower()
                user = action.content.lower()

                if any(word in user for word in expected.split()):
                    reward += 1.0
                else:
                    reward -= 0.5

        # ❌ Penalty
        if self.step_count > 3:
            reward -= 1.0

        done = True

        return (
            Observation(
                email=self.current["text"],
                step=self.step_count,
                task=self.task
            ),
            reward,
            done,
            {}
        )

    def state(self):
        return self.current
