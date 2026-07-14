@echo off
setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

for %%S in (aep comed dayton pjme) do (
  echo Running PJM %%S...
  python scripts\run_posthoc_energy_benchmarks_v4_3.py --project-root . --worker --dataset pjm --series %%S || exit /b 1
)
python scripts\run_posthoc_energy_benchmarks_v4_3.py --project-root . --aggregate-only || exit /b 1
echo Post-hoc SARIMA/ETS benchmark complete.
exit /b 0
