from utils import *
from models.maskedTimeSeriesTransformerWithHistory import MaskedTimeSeriesTransformerWithHistory
import argparse
import pandas as pd
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Compute MSE statistics for multiple context sizes.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/net/tscratch/people/plgfimpro/korelacje/model_with_revin_fixed_context/masked_m1_in100_out50_lr1e-3-0.pt"),
        help="Directory containing the model checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = {}
    for context_size in [300, 200, 100, 50, 25, 10, 1]:
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
                Path("/net/tscratch/people/plgfimpro/korelacje/short_fixed_results_openface2"),
                context_size,
                50,
            )
            torch.save(
                {"X_train": X_train, "Y_train": Y_train, "X_test": X_test, "Y_test": Y_test},
                dataset_path,
            )
        datasets[context_size] = (X_train, Y_train, X_test, Y_test)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskedTimeSeriesTransformerWithHistory(d_model=32).to(device)
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])

    result_df = pd.DataFrame(columns=['context_size', 'mse_no_mask_mom_AU06_infant', 'mse_no_mask_mom_AU12_infant',
                                 'mse_no_mask_mom_AU06_adult', 'mse_no_mask_mom_AU12_adult',
                                 'mse_masked_mom_AU06_infant', 'mse_masked_mom_AU12_infant',
                                 'mse_masked_mom_AU06_adult', 'mse_masked_mom_AU12_adult',
                                 'mse_masked_child_AU06_infant', 'mse_masked_child_AU12_infant',
                                 'mse_masked_child_AU06_adult', 'mse_masked_child_AU12_adult',])

    for context_size, (X_train, Y_train, X_test, Y_test) in datasets.items():
        print(f"Evaluating dataset with {context_size}")
        train_dataset =  TimeSeriesDataset(X_train, Y_train)
        test_dataset = TimeSeriesDataset(X_test, Y_test)
        print(train_dataset, test_dataset)

        num_examples = min(500, len(test_dataset))
        not_masked = calculate_MSE(model, start=0, timestep=9, dataset=test_dataset, num_examples=num_examples)
        masked_mom = calculate_MSE(model, start=0, timestep=9, mask_mom=True, dataset=test_dataset, num_examples=num_examples)
        masked_child = calculate_MSE(model, start=0, timestep=9, mask_child=True, dataset=test_dataset, num_examples=num_examples)
        result_df.loc[len(result_df)] = np.concatenate(([context_size], not_masked, masked_mom, masked_child))

    output_path = Path(f"/net/tscratch/people/plgfimpro/korelacje/time_window_{args.model_path.stem}.csv")
    result_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()