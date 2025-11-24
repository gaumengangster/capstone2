# Base image
FROM python:3.11.13-slim-bookworm

# Set working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Make port 8085 available to the world outside this container
EXPOSE 8085

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8085", "app:app"]