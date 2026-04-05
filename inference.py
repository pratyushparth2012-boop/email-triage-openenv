import asyncio
import os
from typing import List, Optional

from openai import OpenAI
from environment import EmailEnv, Action

# ---------------- ENV VARIABLES ---------------- #

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

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
                {
                    "role": "user",
                    "content": f"Classify or respond to this email: {email}"
                }
            ],
        )

        text = response.choices[0].message.content.strip().lower()
        return text if text else "spam"

    except Exception:
        return "spam"


# ---------------- TASK RUNNER ---------------- #

async def run_task(task_name: str):
    env = EmailEnv(task=task_name)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env="email-env", model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):

            action_text = get_model_output(obs.email)

            action = Action(
                action_type="classify" if task_name == "easy" else "reply",
                content=action_text
            )

            result = await env.step(action)

            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

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

        # ✅ Normalize score between 0 and 1
        total_reward = sum(rewards)

        score = max(0.0, min(total_reward, 1.0))

        success = score >= 0.3

    except Exception as e:
        log_step(
            step=steps_taken,
            action="error",
            reward=0.00,
            done=True,
            error=str(e)
        )
        success = False
        score = 0.0

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )


# ---------------- MAIN ---------------- #

async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())
