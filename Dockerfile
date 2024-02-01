FROM python:3.11-slim as build

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
# The --mount will mount the buildx cache directory to where
# Poetry and Pip store their cache so that they can re-use it

ENV PYTHONUNBUFFERED=1 \
    # pip
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="$POETRY_HOME/bin:$PATH"


RUN apt-get update && \
    apt-get install -y \
    apt-transport-https \
    gnupg \
    ca-certificates \
    build-essential \
    git \
    nano \
    curl

RUN curl -sSL https://install.python-poetry.org | python3 -


# used to init dependencies
COPY poetry.lock pyproject.toml README.md ./
COPY ml_pipelines ./ml_pipelines
RUN PATH=$POETRY_HOME/bin:$PATH poetry build

# # # Stage 2: Production stage
FROM python:3.11-slim

ARG UID=1000
ARG GID=1000
WORKDIR /package

RUN  groupadd -g "${GID}" python \
  && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" python \
  && chown python:python -R /package

# For some reason copying to /package/dist/. doesn't work on AWS
COPY --from=build /dist/*.whl /package/.
RUN pip install *.whl

USER python
