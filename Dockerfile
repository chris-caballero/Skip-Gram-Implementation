FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN python3 -m nltk.downloader stopwords

# Copy the rest of the code
COPY data /app/data
COPY imgs /app/imgs
COPY src /app/src

EXPOSE 8888

# Set environment variables
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=docker

# Specify the command to run when the container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/app/src/notebooks", "--NotebookApp.extra_static_paths=/app/data:/app/imgs"]

