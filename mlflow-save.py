# %%
import os
import mlflow
import json
RESULT_DIR = "./results"

# %%

# update these lines
EXP_NAME = "exp40"
PUBLIC_SCORE = 0.899

# %%
mlflow.start_run(run_name=EXP_NAME)

# %%
with open(os.path.join(RESULT_DIR, EXP_NAME, "configuration.json")) as f:
    CFG = json.load(f)

with open(os.path.join(RESULT_DIR, EXP_NAME, "result.json")) as f:
    RESULT = json.load(f)

RESULT.update({"public_score": PUBLIC_SCORE})

for k, v in CFG.items():
    mlflow.log_param(k, v)
for k, v in RESULT.items():
    mlflow.log_metric(k, v)

# %%
mlflow.end_run()

# %%
