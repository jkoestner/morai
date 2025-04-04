# this docker file is used to create a docker image for the web.
# It currently is being built on dockerhub at dmbymdt/morai and
# then pulled down into a web container.
# To run dockerfile and create own image use from where the dockerfile is located.:
#   `docker build --no-cache -t morai .` 
# If wanting to build from a specific branch use:
#   `docker build --build-arg BRANCH_NAME=dev --no-cache -t morai .`
#
# slim was used instead of alpine because of the need of numpy
FROM python:3.12-slim

# Install dependencies, git
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set work directory
WORKDIR /code

# Define a build-time argument for the branch name
ARG BRANCH_NAME=main

# Install the package from a specific branch
RUN uv pip install --no-cache-dir --system "git+https://github.com/jkoestner/morai.git@${BRANCH_NAME}"

# Create new user
RUN adduser --disabled-password --gecos '' morai && \
    chown -R morai:morai /code 
USER morai

# Using port 8001 for web
EXPOSE 8001