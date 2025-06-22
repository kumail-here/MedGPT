# Base Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Fix Streamlit permission error
ENV STREAMLIT_HOME /app/.streamlit
RUN mkdir -p /app/.streamlit

# Streamlit runs on port 7860
EXPOSE 7860

# Launch the Streamlit app
CMD ["streamlit", "run", "app/main.py", "--server.port=7860", "--server.address=0.0.0.0"]
