import typer

from ml_pipelines.deployment.local.eval import main as eval_local
from ml_pipelines.deployment.local.serve import main as serve_local
from ml_pipelines.deployment.local.train import main as train_local

app = typer.Typer()

app.command("train_local")(train_local)
app.command("eval_local")(eval_local)
app.command("serve_local")(serve_local)

app()
