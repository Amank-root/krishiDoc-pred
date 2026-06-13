FROM python:3.11-slim

WORKDIR /app

# Step 1: Install curl safely and clean up the package cache to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Step 2: Download and install uv using the official Astral script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Step 3: Explicitly add uv to the system PATH so Docker can find it
ENV PATH="/root/.local/bin/:$PATH"

# Step 4: Copy project files and synchronize dependencies
COPY . .
RUN uv sync 

# Step 5: Execute the FastAPI/Uvicorn app through the uv environment wrapper
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
