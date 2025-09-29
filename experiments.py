import sys
import traceback

from crslab.config import Config
from crslab.quick_start import run_crslab

dataset = 'redial'

# List of configs to run
config_list = [
    f"config/conversation/gpt2/{dataset}.yaml",
    f"config/crs/redial/{dataset}.yaml",
    f"config/crs/inspired/{dataset}.yaml",
    f"config/crs/kbrd/{dataset}.yaml",
    f"config/recommendation/bert/{dataset}.yaml",
    f"config/recommendation/gru4rec/{dataset}.yaml",
    f"config/recommendation/popularity/{dataset}.yaml",
    f"config/recommendation/sasrec/{dataset}.yaml",
]

error_log_file = 'ERROR_LOG.TXT'

for config_path in config_list:
    print(f"\n=== Running config: {config_path} ===\n")
    try:
        # Simulate command-line args
        gpu = '7'
        debug = False

        # Set up configuration
        config = Config(config_path, gpu, debug)

        # Run the system
        run_crslab(config,
                   save_data=False,
                   restore_data=False,
                   save_system=False,
                   restore_system=False,
                   interact=False,
                   debug=debug,
                   tensorboard=False)

        print(f"\n+++ Finished successfully: {config_path} +++\n")

    except Exception as e:
        print(f"\n--- Error running config: {config_path} ---\n")
        with open(error_log_file, 'a') as f:
            f.write(f"Error with config: {config_path}\n")
            f.write("Exception Traceback:\n")
            traceback.print_exc(file=f)
            f.write("\n" + "="*80 + "\n")
