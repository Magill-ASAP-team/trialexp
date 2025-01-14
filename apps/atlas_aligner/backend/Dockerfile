FROM python:3.9

WORKDIR /app

# Install uv.
RUN apt-get update && apt-get install -y git
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# cache the project dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy the startup script into the container
COPY startup.sh /usr/local/bin/startup.sh
# Ensure the startup script is executable
RUN chmod +x /usr/local/bin/startup.sh



CMD ["/usr/local/bin/startup.sh"]