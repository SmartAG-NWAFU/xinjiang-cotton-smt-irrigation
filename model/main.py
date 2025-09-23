from site_deficit_irrigate_optimate import run_optimize
from simulation import convent_simulate, deficit_simulate, expert_simulate, future_simulate

# start optimize deficit thresholds
run_optimize()

# start deficit simulate
deficit_simulate()