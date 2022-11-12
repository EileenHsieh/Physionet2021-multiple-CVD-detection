#%%
import mlflow as mf
from mlflow.tracking.client import MlflowClient
from pathlib import Path
from copy import deepcopy
from mlflow.entities import Param
#%%
MLRUNS_DIR = str(Path("/homes/ssd0/chadyang/Projects/Challenges/Physionet/cinc/scripts/PhysioNet-CinC-Challenges-2021/mlruns"))
EXP_NAME = "tune-tfm-raw-bce"

#%%
client = MlflowClient(tracking_uri=MLRUNS_DIR)
exp = client.get_experiment_by_name(EXP_NAME)
exp_id = exp.experiment_id

#%%
run_infos = client.list_run_infos(experiment_id=exp_id)
for run_info in run_infos:
    print("- run_id: {}, lifecycle_stage: {}".format(run_info.run_id, run_info.lifecycle_stage))
    run = client.get_run(run_info.run_id)
    params = run.data.params
    # params_fake = deepcopy(params)
    if params_fake == params:
        print(run.data.metrics["cv_test_cm"])

    del params_fake["verbose"]
    assert params_fake==params
    








