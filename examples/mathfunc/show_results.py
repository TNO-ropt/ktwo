# ruff: noqa
from ropt.results import FunctionResults

results = FunctionResults.from_netcdf("everest_output/results/result00.nc")
print(results.to_dataframe("evaluations"))
