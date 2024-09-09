from argparse import ArgumentParser, Namespace
from sklearn.model_selection import train_test_split
import torch
from predictor_s import DemographicPredictor
from predictor_y import Classifier
from data_module import DataModelMissingSensitiveAtt, BaseDataModule
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import NeptuneLogger
import torch
from datasets import get_datasets_tp
import numpy as np
from sklearn.metrics import accuracy_score, auc
from knn_imputer import knn_impute
from ARL.arl import ARL
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
)
from metrics import equal_opportunity
import pandas as pd
import os
from pytorch_lightning.callbacks import EarlyStopping
from FAIRDA.fair_da import FairDA

parser = ArgumentParser()
parser.add_argument("--devices", type=int, default=1, help="number of GPUs/CPUs")
parser.add_argument("--accelerator", type=str, default="cpu", help="Device type")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--fast_dev_run", type=int, default=0, help="Fast check")
parser.add_argument("--num_epoch", type=int, default=500, help="Number of epoch")
parser.add_argument(
    "--dataset",
    type=str,
    default="adult",
    help="3 datasets are available: adult, new_adult, acs_employment",
)

parser.add_argument(
    "--sensitive_feature_type",
    type=str,
    default="clean",
    help="Use clean or predicted sensitive features. `clean`, `ours` or `predicted`",
)

parser.add_argument("--debug", action="store_true", help="Log debugs")
parser.add_argument(
    "--demographic_predictor",
    type=str,
    default="DNN",
    help="Model used to infer the sensitive attribute from the related feature. `DNN`, `KNN`",
)

parser.add_argument(
    "--baseline",
    type=str,
    default="VANILLA",
    help="the baseline model: `ARL`, `DRO`, `FAIR_BATCH`, `VANILLA` ",
)

parser.add_argument(
    "--target_fairness",
    type=str,
    default="dp",
    help="the baseline model: `eqopp`, `eqodds`, `dp`",
)
parser.add_argument("--num_workers", type=int, default=0, help="Number of epoch")
parser.add_argument("--seed", type=int, default=1, help="Number of seeds")


arg = parser.parse_args()

torch.manual_seed(arg.seed)
np.random.seed(arg.seed)
args = {
    "num_epoch": arg.num_epoch,
    "devices": arg.devices,
    "accelerator": arg.accelerator,
    "b1": 0.5,
    "b2": 0.999,
    "fair_batch_params": {"target_fairness": arg.target_fairness},
}
datasets = get_datasets_tp()

fair_metrics_map = {
    "dp": demographic_parity_difference,
    "eop": equal_opportunity,
    "eodds": equalized_odds_difference,
}


def train_and_predict(train_data, test_data):
    n_features = train_data[0].shape[1]
    if arg.baseline == "ARL":
        cls = ARL(
            input_size=n_features,
            lr=0.001,
            betas=(args["b1"], args["b2"]),
            batch_size=arg.batch_size,
            pretrain_steps=(arg.num_epoch // 10),
        )
    elif arg.baseline == "DRO":
        cls = Classifier(
            input_size=n_features,
            lr=0.001,
            betas=(args["b1"], args["b2"]),
            use_robust=True,
        )
    elif arg.baseline == "CVAR":
        cls = Classifier(
            input_size=n_features,
            lr=0.001,
            betas=(args["b1"], args["b2"]),
            use_robust=True,
            robust_method="cvar",
        )
    elif arg.baseline == "VANILLA" or arg.baseline == "FAIR_BATCH":
        cls = Classifier(
            input_size=n_features, lr=0.001, betas=(args["b1"], args["b2"])
        )
    elif arg.baseline == "FAIRDA":
        cls = FairDA(
            input_size=n_features, output_size=1, lr=0.001, labelled_bs=arg.batch_size
        )
    else:
        cls = None
        print("UNKNOWN BASELINE")
        return
    data_module = BaseDataModule(
        train_data=train_data,
        test_data=test_data,
        model=None,
        batch_size=arg.batch_size,
        n_features=n_features,
        num_workers=arg.num_workers,
        use_validation=True,
    )
    if arg.baseline == "FAIR_BATCH":
        data_module = BaseDataModule(
            train_data=train_data,
            test_data=test_data,
            model=cls.model,
            batch_size=arg.batch_size,
            n_features=n_features,
            num_workers=arg.num_workers,
            use_fair_batch=True,
            fair_batch_params=args["fair_batch_params"],
        )
    elif arg.baseline == "FAIRDA":
        data_module = DataModelMissingSensitiveAtt(
            data1=data1,
            data2=data2,
            data1_test=test_data,
            batch_size=arg.batch_size * 2,
            val_size=0.1,
            num_workers=arg.num_workers,
            include_y_in_x=False,
            labeled_bs=arg.batch_size,
        )
    X_test, y_test, s_test = test_data
    # specify token and project to use NeptuneLogger set it to none
    if arg.debug:
        """neptune_logger = NeptuneLogger(
            api_token=YOUR_TOKEN,
            project=YOUR_PROJECT,
            tags=[arg.dataset, "target_predictor"],
        )"""
    else:
        neptune_logger = None

    early_stop_callback = EarlyStopping(
        monitor="acc/train", min_delta=0.00, patience=5, verbose=False, mode="max"
    )
    trainer = Trainer(
        devices=arg.devices,
        accelerator=arg.accelerator,
        enable_progress_bar=arg.debug,
        max_epochs=arg.num_epoch,
        logger=neptune_logger,
        fast_dev_run=arg.fast_dev_run,
    )
    trainer.fit(cls, datamodule=data_module)
    # trainer.test(cls, datamodule=data_module)
    if neptune_logger:
        trainer.logger.log_hyperparams(arg)

    # print(">>>>", X_test.shape, n_features)
    y_pred = cls.predict(torch.from_numpy(X_test).float()).detach().cpu().numpy()

    results = {}
    results["acc"] = accuracy_score(y_test, y_pred)

    for fair_metric in fair_metrics_map:
        results[fair_metric] = [
            fair_metrics_map[fair_metric](y_test, y_pred, sensitive_features=s_test)
        ]

    idx_0 = np.where(s_test == 0)
    idx_1 = np.where(s_test == 1)

    results["acc_0"] = [accuracy_score(y_test[idx_0], y_pred[idx_0])]
    results["acc_1"] = [accuracy_score(y_test[idx_1], y_pred[idx_1])]
    results["count_0"] = [len(y_test[idx_0]) / len(y_test)]
    results["count_1"] = [len(y_test[idx_1]) / len(y_test)]

    print(results)

    out_path = "outputs/baselines/{}/{}".format(arg.baseline, arg.dataset)
    out_file = "{}/baseline_{}{}_{}.csv".format(
        out_path,
        arg.dataset,
        "_{}".format(arg.target_fairness) if arg.baseline == "FAIR_BATCH" else "",
        arg.seed,
    )
    print(out_file)

    os.makedirs(out_path, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_file, encoding="utf-8", index=False)

    # return trainer.predict(demp, dataloaders=data_module.test_dataloader(), return_predictions=True)


data1, data2 = datasets[arg.dataset]()
print(
    "===================================== Running baseline={} seed={} ============ ".format(
        arg.baseline, arg.seed
    )
)
X, y, s = data1

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, s, test_size=0.3, random_state=arg.seed
)


results = train_and_predict(
    train_data=(X_train, y_train, s_train), test_data=(X_test, y_test, s_test)
)

print("===================================== Completed")
