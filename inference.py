import os
from openai import OpenAI
from environment import EmailEnv, Action

client = OpenAI(base_url=os.getenv("API_BASE_URL"))
MODEL_NAME = os.getenv("MODEL_NAME")

for task in ["easy", "medium", "hard"]:
    print(f"\nRunning task: {task}")

    env = EmailEnv(task=task)
    obs = env.reset()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": f"Email: {obs.email}. What should I do?"}
        ]
    )

    action = Action(
        action_type="reply" if task != "easy" else "classify",
        content=response.choices[0].message.content.strip().lower()
    )

    obs, reward, done, _ = env.step(action)

    print("Reward:", reward)
