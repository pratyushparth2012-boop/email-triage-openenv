import asyncio
import os
from typing import List, Optional

from openai import OpenAI
from environment import EmailEnv, Action

# ✅ ENV VARIABLES (ONLY REQUIRED ONES)
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# ❌ DO NOT USE HF_TOKEN (as per checklist)
client = OpenAI(base_url=API_BASE_URL)

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 3


# ---------------- LOGGING ---------------- #

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------- MODEL CALL ---------------- #

def get_model_output(email: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": f"Process this email: {email}"}
            ],
        )
        return response.choices[0].message.content.strip().lower()
    except Exception:
        return "spam"


# ---------------- MAIN LOOP ---------------- #

async def run_task(task_name: str):
    env = EmailEnv(task=task_name)

    rewards = []
    steps_taken = 0

    log_start(task=task_name, env="email-env", model=MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            action_text = get_model_output(obs.email)

            action = Action(
                action_type="classify" if task_name == "easy" else "reply",
                content=action_text
            )

            obs, reward, done, _ = env.step(action)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_text,
                reward=reward,
                done=done,
                error=None
            )

            if done:
                break

        # ✅ SCORE NORMALIZATION
        score = sum(rewards)
        score = max(0.0, min(score, 1.0))

        success = score > 0.3

    except Exception as e:
        log_step(step=steps_taken, action="error", reward=0.0, done=True, error=str(e))
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------- ENTRY POINT ---------------- #

async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())
