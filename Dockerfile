# Stage 1: Build environment
FROM python:3.9 AS build

# Set the working directory in the container
WORKDIR /app

# Create the /app directory and copy the requirements file into the container
RUN mkdir /app
COPY requirements.txt /app/

# Install project dependencies to a temporary location
RUN pip install --user -r /app/requirements.txt

# Stage 2: Final image
FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the project files from the build stage
COPY . .

# Copy the installed dependencies from the build stage
COPY --from=build /root/.local /root/.local

# Update PATH to include user-installed Python dependencies
ENV PATH=/root/.local/bin:$PATH

# Start your application here (e.g., run your Python script)
CMD ["python", "main.py"]
