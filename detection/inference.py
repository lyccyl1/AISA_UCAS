import os
import argparse
import pandas as pd

from utils import FolderDataset
from utils import Runner

def get_opts():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--your-team-name",
        type=str,
    )
    arg.add_argument(
        "--data-folder",
        type=str,
    )
    arg.add_argument(
        "--model-weights",
        type=str,
    )
    arg.add_argument(
        "--result-path",
        type=str,
    )
    opts = arg.parse_args()
    return opts


def get_dataset(opts):
    ### tips: customize your transforms
    import torchvision.transforms as Transforms
    transforms = Transforms.Compose(
        [
            Transforms.Resize((299, 299)),
            Transforms.ToTensor(),
            Transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]
    )

    # DO NOT change FolderDataset
    return FolderDataset(opts.data_folder, transforms) 


def get_model_runner(opts, dataset):
    ### tips: customize your model
    import torch
    from utils import Xception
    model = Xception()
    model.fc = torch.nn.Linear(2048, 1)
    model.load_state_dict(
        torch.load(opts.model_weights)
    )

    # DO NOT change Runner
    runner = Runner(model, dataset)
    return runner


if __name__ == "__main__":
    opts = get_opts()
    dataset = get_dataset(opts)
    runner = get_model_runner(opts, dataset)
    results = runner.run()

    os.makedirs(opts.result_path, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(opts.result_path, opts.your_team_name + ".xlsx"))
    prediction_frame = pd.DataFrame(
        data = {
            "img_names": results["predictions"].keys(),
            "predictions": results["predictions"].values(),
        }
    )
    time_frame = pd.DataFrame(
        data = {
            "Data Volume": [len(results["predictions"].keys())],
            "Time": [results["time"]],
        }
    )
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()