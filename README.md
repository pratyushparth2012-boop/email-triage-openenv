# 📧 Email Triage OpenEnv Environment

## 🧠 Overview

This project implements a real-world task simulation using the OpenEnv framework.  
The environment simulates an email triage and response system, where an AI agent must:

- Classify emails (spam or important)
- Decide whether a reply is needed
- Generate appropriate responses

---

## 🎯 Motivation

Email management is a common real-world task. This environment helps evaluate how well AI agents can:

- Understand intent
- Make decisions
- Generate useful responses

---

## ⚙️ Environment Design

### 🔹 Observation Space

{
  "email": str,
  "step": int,
  "task": str
}

- email: Input email text  
- step: Current step count  
- task: Task difficulty (easy / medium / hard)  

---

### 🔹 Action Space

{
  "action_type": str,
  "content": str
}

- action_type: "classify" or "reply"  
- content: Classification label or reply text  

---

### 🔹 Reward Function

Easy Task:
- Correct classification: +1.0  
- Wrong classification: -1.0  

Medium Task:
- Correct classification: +0.5  
- Valid reply: +0.5  

Hard Task:
- Relevant reply: up to +1.0  
- Irrelevant reply: penalty  

Penalty:
- Too many steps: -1.0  

---

## 🧪 Tasks

### Easy
- Classify email as spam or important

### Medium
- Classify email  
- Generate a simple reply  

### Hard
- Generate a context-aware response  
- Graded using keyword matching  

---

## 🤖 Baseline Inference

The baseline agent uses the OpenAI API.

Environment variables required:

export API_BASE_URL="your_api_url"  
export MODEL_NAME="gpt-4o-mini"  

Run:

python inference.py

---

## 📦 Project Structure

openenv-email/

- environment.py  
- inference.py  
- grader.py  
- openenv.yaml  
- Dockerfile  
- requirements.txt  
- README.md  

---

## 🐳 Docker Setup

Build:

docker build -t email-env .

Run:

docker run email-env

---

## 🚀 Deployment

This project is designed for Hugging Face Spaces with containerized execution and OpenEnv compatibility.

---

## 📊 Baseline Results

Easy:   0.5–1.0  
Medium: 0.5–0.8  
Hard:   0.3–0.7  

---

## ✅ Features

- Real-world task simulation  
- OpenEnv compliant structure  
- Multi-task evaluation  
- Reward shaping  
- Dockerized setup  

---

## ⚠️ Notes

- Only required environment variables are used:
  API_BASE_URL  
  MODEL_NAME  

- No unnecessary tokens or configs  
- Designed to pass openenv validate checks  
