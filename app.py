from fastapi import FastAPI
from environment import EmailEnv, Action
from pydantic import BaseModel

app = FastAPI()

env = EmailEnv()


class StepRequest(BaseModel):
    action_type: str
    content: str


@app.post("/reset")
async def reset():
    result = await env.reset()
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }


@app.post("/step")
async def step(req: StepRequest):
    action = Action(
        action_type=req.action_type,
        content=req.content
    )

    result = await env.step(action)

    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }
