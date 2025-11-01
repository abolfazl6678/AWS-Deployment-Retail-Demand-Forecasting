# Base image with Python
FROM python:3.10-slim
# Metadata
LABEL maintainer="Abolfazl Zolfaghari Email: ab.zolfaghari.abbasghaleh@gmail.com "
LABEL version="1.0"
LABEL description="FastAPI Demand Forecasting API"
# Set working directory
WORKDIR /app
# Copy requirements first (leverages Docker cache)
COPY requirements.txt .
# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt
# Copy rest of the project 
COPY . .
# Set port to be used
EXPOSE 8000
# initial running programs
CMD ["uvicorn", "Api_tensor_flow:app", "--host", "0.0.0.0", "--port", "8000"]