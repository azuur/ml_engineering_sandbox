# ML pipelines

Simple ML pipeline repo for experimenting with CI/CD / DevOps / MLOps.

## To do

~~- Use Python 3.11 in CI github action~~
~~- make pipelines functions~~
~~- add loggers to stuff~~
~~- add local deployment code...~~
~~- add versioning to training... in deployment?~~
~~- add eval pipeline, model comparison~~
~~- add "best model" mark. add "get_best_model"~~
~~- add Dockerfile~~
- add real prediction logging func
- add db conn / func to save inference cases (local deployment)
- add build script to push to ECR (AWS deployment)
- add rest of AWS deployment (using S3, EC2, AWS CodePipeline)

# Commands to remember
- python ml_pipelines/deployment/local/train.py
- python ml_pipelines/deployment/local/eval.py
- python ml_pipelines/deployment/local/serve.py
- sudo docker build -t ml_pipelines:latest .
- sudo docker run --rm -it ml_pipelines:latest /bin/sh
- sudo docker run --rm -it ml_pipelines:latest 'python ml_pipelines/deployment/local/train.py'
