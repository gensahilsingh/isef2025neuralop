"""Minimal training scaffold for the cardiac-tuned FNO (FNOCardiac).

This script demonstrates how to instantiate the model from
`config.cardiac_config.Default` and print model summary. It can be
expanded to a full training loop following patterns in other scripts.
"""

import sys

from zencfg import make_config_from_cli

from neuralop.training import setup
from neuralop.utpip install -e .
ils import count_model_params, get_project_root
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import get_model

sys.path.insert(0, "../")
from config.cardiac_config import Default


def main():
    config = make_config_from_cli(Default)
    config = config.to_dict()

    device, is_logger = setup(config)

    model = get_model(config)

    if is_logger:
        print("Model:")
        print(model)
        print(f"n_params: {count_model_params(model)}")

    # Example: create optimizer (expand as needed)
    optimizer = AdamW(model.parameters(), lr=config.opt.learning_rate, weight_decay=config.opt.weight_decay)

    # Placeholder: wiring up dataset and trainer follows existing training scripts
    print("Cardiac FNO scaffold ready. Plug in data loader and Trainer to start training.")


if __name__ == "__main__":
    main()
