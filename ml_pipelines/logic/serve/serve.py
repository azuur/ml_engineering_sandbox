def validate_sample(sample: dict):
    x1_upper_bound = 5
    if sample["X1"] > x1_upper_bound:
        return False
    return True
