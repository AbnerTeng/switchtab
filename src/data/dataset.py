import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset


def load_training_data(
    data_name: str,
    data_seed: int,
    batch_size: int,
    data_dir: str,
    early_stopping: bool = False,
):
    with open(f"{data_dir}/{data_name}/{data_seed}.pkl", "rb") as f:
        data_dict = pickle.load(f)

        tr_x = torch.tensor(data_dict["x_train"], dtype=torch.float32)
        tr_y = torch.tensor(data_dict["y_train"], dtype=torch.float32)

        if early_stopping:
            print("!!! split validation set !!!")
            tr_count = int(tr_x.shape[0] * 0.8)
            va_x = tr_x[tr_count:]
            va_y = tr_y[tr_count:]
            tr_x = tr_x[:tr_count]
            tr_y = tr_y[:tr_count]
            va_dataset = TensorDataset(va_x, va_y)
            va_dataloader = DataLoader(
                va_dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )

        else:
            va_dataloader = None

        tr_dataset = TensorDataset(tr_x, tr_y)
        tr_dataloader = DataLoader(
            tr_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        return (
            tr_dataloader,
            va_dataloader,
            data_dict["col_cat_count"],
            data_dict["label_cat_count"],
        )


def load_valid_test_data(
    data_name: str,
    data_seed: int,
    batch_size: int,
    target_transform: bool,
    data_dir: str,
    verbose: bool = True,
):
    with open(f"{data_dir}/{data_name}/{data_seed}.pkl", "rb") as f:
        data_dict = pickle.load(f)

    if verbose:
        print(f"> val datasize: {data_dict['x_val'].shape}")
        print(f"> test datasize: {data_dict['x_test'].shape}")
        print(
            f"> estimated full datasize: {int((data_dict['x_test'].shape[0] + data_dict['x_val'].shape[0]) / 0.3)}"
        )

    valid_dataset = TensorDataset(
        torch.tensor(data_dict["x_val"]),
        torch.tensor(data_dict["y_val" if not target_transform else "y_val_transform"]),
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_dataset = TensorDataset(
        torch.tensor(data_dict["x_test"]),
        torch.tensor(
            data_dict["y_test" if not target_transform else "y_test_transform"]
        ),
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return (
        valid_dataloader,
        test_dataloader,
        data_dict["target_transformer"],
        data_dict["dataset_config"]["regression"],
    )
