# ML pipelines

Simple ML pipeline repo for experimenting with CI/CD / DevOps / MLOps.

## The idea

This repo contains code to train, evaluate, and serve a simple machine learning model. The objective of this project is to work on my MLOps and ML engineering skills.

Some ideas I am implementing in this repo are:

- Do things as simply but professionally as possible. A simple working solution is better than a sophisticated solution that isn't deployed. ([Agile is the only thing that works](https://www.youtube.com/watch?v=9K20e7jlQPA).)
- [Oneflow](https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow) as the Git branching strategy.
- Cleanish architecture. For now, this means that code is separated between "core" and "non-core" tasks and data structures, and "core" code doesn't depend on "non-core" code.
- One repo and one docker image for train, eval, and serve. IMO, this makes sharing functionality across tasks easier and artifact versioning simpler. (But I'm interested in hearing about drawbacks, too.)
- Use Python standard tooling to make collaboration easier. In particular, the project is a Python package with [poetry](https://python-poetry.org/) as the build backend.
- CI is done using [pre-commit](https://pre-commit.com/) and GitHub actions (since we're in GitHub).
- CD should be done depending on how the project is to be deployed. Currently, I'm experimenting with AWS for deployment, so I also use it for CD.

Since the point of this project is _not_ to sharpen my data anaylsis/science skills, the actual data for the project is completely simulated. Maybe later I will try to modify this in order to actually solve a useful problem.

## To do

- Add section detailing v1 CD + deployment on AWS (with CodeBuild, ECR, Fargate ECS tasks and services, and ELB).
- Create deployment stack using IaC tool (could be AWS CloudFormation).
- Add real prediction logging func
- Add simple demo unit tests
- Add db conn / func to save inference cases
- Add build script to push to ECR (AWS deployment)

# Commands to remember

This is a bit inelegant. Sorry.

- python ml_pipelines/deployment/local/train.py
- python ml_pipelines/deployment/local/eval.py
- python ml_pipelines/deployment/local/serve.py
- python -m ml_pipelines train_local
- python -m ml_pipelines eval_local
- python -m ml_pipelines serve_local
- sudo docker build -t ml_pipelines:latest .
- sudo docker run --rm -it ml_pipelines:latest /bin/sh
- sudo docker run --rm -it -v localpath:containerpath --env-file .env ml_pipelines:latest /bin/bash -c "python -m ml_pipelines serve_local"
- sudo docker run --rm -it -v localpath:containerpath --env-file .env --network host ml_pipelines:latest /bin/bash -c "python -m ml_pipelines serve_local"
