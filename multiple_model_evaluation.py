from pathlib import Path
import argparse
import pandas as pd
import torch
from models.maskedTimeSeriesTransformerWithHistory import MaskedTimeSeriesTransformerWithHistory as MaskedTimeSeriesTransformer
import numpy as np
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple models across cached datasets.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/net/tscratch/people/plgfimpro/korelacje/short_fixed_results_openface2"),
        help="Root directory of the raw dataset.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("/net/tscratch/people/plgfimpro/korelacje/model_with_revin_fixed_context"),
        help="Directory containing model checkpoints.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = {}
    for context_size in [300, 100]:
        print("loaded dataset {context_size}".format(context_size=context_size))
        dataset_path = Path(f"datasets/dataset_ctx{context_size}_out50.pt")
        if dataset_path.exists():
            data = torch.load(dataset_path, map_location="cpu")
            X_train = data["X_train"]
            Y_train = data["Y_train"]
            X_test = data["X_test"]
            Y_test = data["Y_test"]
        else:
            X_train, Y_train, X_test, Y_test = load_multiple_files(
                2,
                args.dataset,
                context_size,
                50,
            )
            torch.save(
                {"X_train": X_train, "Y_train": Y_train, "X_test": X_test, "Y_test": Y_test},
                dataset_path,
            )
        datasets[context_size] = (X_train, Y_train, X_test, Y_test)

    result_df = pd.DataFrame(columns=[
        'model_name',
        'context_size',
        'mse_no_mask_mom_AU06_infant', 'mse_no_mask_mom_AU12_infant',
        'mse_no_mask_mom_AU06_adult', 'mse_no_mask_mom_AU12_adult',
        'mse_masked_mom_AU06_infant', 'mse_masked_mom_AU12_infant',
        'mse_masked_mom_AU06_adult', 'mse_masked_mom_AU12_adult',
        'mse_masked_child_AU06_infant', 'mse_masked_child_AU12_infant',
        'mse_masked_child_AU06_adult', 'mse_masked_child_AU12_adult',
    ])

    for modelpath in sorted(args.dir.glob('*_lr1e-3-?.pt')):
        print(f"Evaluating model {modelpath.name}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MaskedTimeSeriesTransformer(d_model=32).to(device)
        model.load_state_dict(torch.load(modelpath)['model_state_dict'])

        for context_size, (X_train, Y_train, X_test, Y_test) in datasets.items():
            test_dataset = TimeSeriesDataset(X_test, Y_test)
            num_examples = min(500, len(test_dataset))
            not_masked = calculate_MSE(model, start=0, timestep=9, dataset=test_dataset, num_examples=num_examples)
            masked_mom = calculate_MSE(model, start=0, timestep=9, mask_mom=True, dataset=test_dataset, num_examples=num_examples)
            masked_child = calculate_MSE(model, start=0, timestep=9, mask_child=True, dataset=test_dataset, num_examples=num_examples)
            result_df.loc[len(result_df)] = np.concatenate((
                [modelpath.name, context_size],
                not_masked,
                masked_mom,
                masked_child,
            ))

        print(modelpath)

    output_path = Path(f"multiple_model_eval_{args.dir.name}.csv")
    result_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()