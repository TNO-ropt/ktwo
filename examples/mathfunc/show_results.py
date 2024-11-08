import pickle  # noqa: D100, INP001
from pathlib import Path

from ropt.config.enopt import EnOptConfig
from ropt.results import FunctionResults

results = FunctionResults.from_netcdf("everest_output/results/result00.nc")
with Path.open("everest_output/results/config.pickle", "rb") as file_obj:
    config_dict = pickle.load(file_obj)  # noqa: S301
config = EnOptConfig.model_validate(config_dict["ropt_config"])
print(results.to_dataframe(config, "evaluations"))
