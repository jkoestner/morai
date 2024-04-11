# this docker file is used to create a docker image for the web.
# It currently is being built on dockerhub at dmbymdt/morai and
# then pulled down into a web container.
# To run dockerfile and create own image `docker build --no-cache -t morai .` 
# from where the dockerfile is located.
FROM python:3.9-slim

# Install git
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /code

# Install requirements (without copying the whole directory)
RUN pip install --no-cache-dir "git+https://github.com/jkoestner/morai.git@main"

# Create new user
RUN adduser --disabled-password --gecos '' morai && \
    chown -R morai:morai /code 
USER morai

# Using port 8001 for web
EXPOSE 8001