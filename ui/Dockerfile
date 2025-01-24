# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set labels
LABEL org.opencontainers.image.authors="Matteo Gianferrari"
LABEL org.opencontainers.image.title="homelens-ca-ui"
LABEL org.opencontainers.image.description="HomeLens CA UI."
LABEL org.opencontainers.image.source=https://github.com/matteogianferrari/homelens-ca

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory to /app
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app

# Install project dependencies
RUN pip install --upgrade --no-cache-dir pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY ./ui /app

# Expose the port your service runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]
