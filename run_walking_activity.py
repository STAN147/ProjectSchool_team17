"""Walking activity: the benchmark config, ModernNCA and CatBoost.

python -u run_walking_activity.py --gpu 0
"""

import sys

from run_benchmark import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--user", "user4",
        "--datasets", "walking-activity",
        "--models", "modernnca", "catboost",
        "--results-dir", "results/walking-activity",
    ]
    main()
