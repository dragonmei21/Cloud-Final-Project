# Final Exam — Theoretical Questions

## 3.1 API Design

### Question 1
The 7 fields are: `piece_id`, `die_matrix`, `lifetime_2nd_strike_s`, `lifetime_3rd_strike_s`, `lifetime_4th_strike_s`, `lifetime_auxiliary_press_s`, and `lifetime_bath_s`. The PLC sends one running timer from the furnace, so cumulative times are the natural data to use. We need all five cumulative timestamps because each partial segment time is calculated from them (the first segment uses only lifetime_2nd_strike_s, the rest use two adjacent values). If any cumulative time is missing, the partial time becomes null, which the response requires.

### Question 2
It’s better to load the file once because reading it on every request is slow and wastes resources. The file never changes while the container is running, so re‑reading it gives no benefits. Loading it once keeps requests fast.

## 3.2 Containerization And Deployment

### Question 1
We use python:3.13-slim because the exam requires Python 3.13 and it’s light.
WORKDIR /app sets the working folder in the container.
pip install uv installs the package manager.
We copy pyproject.toml and uv.lock first so dependency install can be cached.
We copy src/ and reference_times.json so the app code and data are inside the image.
uv sync --frozen installs exact locked dependencies.
EXPOSE 8000 shows which port the app uses.
The CMD runs FastAPI with Uvicorn.

### Question 2
AWS Lambda: Good because you don’t manage servers and it can scale to zero. Bad because cold starts and size limits can hurt FastAPI app.
EC2: Good because you control the server and can tune it. Bad because you must manage updates, scaling, and uptime yourself.

## 3.3 Testing And Extensibility

### Question 1
Testing the pure function is faster and more reliable. You don’t need to start a server or deal with network issues. It checks the logic directly and makes the tests consistent and easy to debug.

### Question 2
Data: add new medians for 6001 in api/reference_times.json.
Code: no change if code loads the JSON dynamically.
Tests: add 6 new tests for matrix 6001.
Deploy: rebuild the Docker image, push to ECR, register a new task definition, and redeploy ECS
