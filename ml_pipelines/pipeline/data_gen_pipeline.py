from ml_pipelines.core.common.dgp import generate_raw_data

data = generate_raw_data(10_000, 813)
data.to_csv("raw_data.csv", index=False)
