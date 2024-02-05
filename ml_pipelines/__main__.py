import typer

from ml_pipelines.deployment.aws.eval import main as eval_aws
from ml_pipelines.deployment.aws.serve import main as serve_aws
from ml_pipelines.deployment.aws.train import main as train_aws
from ml_pipelines.deployment.local.eval import main as eval_local
from ml_pipelines.deployment.local.serve import main as serve_local
from ml_pipelines.deployment.local.train import main as train_local

app = typer.Typer()

app.command("train_local")(train_local)
app.command("eval_local")(eval_local)
app.command("serve_local")(serve_local)
app.command("train_aws")(train_aws)
app.command("eval_aws")(eval_aws)
app.command("serve_aws")(serve_aws)

app()
