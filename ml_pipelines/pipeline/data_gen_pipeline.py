from ml_pipelines.logic.common.dgp import generate_raw_data

data = generate_raw_data(10_000, 813)
data.to_csv("raw_data.csv", index=False)
