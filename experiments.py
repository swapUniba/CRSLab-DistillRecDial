import sys
import traceback

from crslab.config import Config
from crslab.quick_start import run_crslab

# List of configs to run
config_list = [
    "config/conversation/gpt2/distillrecdial.yaml",
    "config/crs/redial/distillrecdial.yaml",
    "config/crs/inspired/distillrecdial.yaml",
    "config/crs/kbrd/distillrecdial.yaml",
    "config/recommendation/bert/distillrecdial.yaml",
    "config/recommendation/gru4rec/distillrecdial.yaml",
    "config/recommendation/popularity/distillrecdial.yaml",
    "config/recommendation/sasrec/distillrecdial.yaml",
]

error_log_file = 'ERROR_LOG.TXT'

for config_path in config_list:
    print(f"\n=== Running config: {config_path} ===\n")
    try:
        # Simulate command-line args
        gpu = '0'
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
