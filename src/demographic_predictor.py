from argparse import ArgumentParser, Namespace
from sklearn.model_selection import train_test_split
import torch
from datasets import get_adult
from predictor_s import DemographicPredictor
from data_module import DataModelMissingSensitiveAtt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import NeptuneLogger
import torch
from datasets import get_old_adult, get_adult, get_acs_employment, get_compas, get_compas_race


parser = ArgumentParser()
parser.add_argument("--devices", type=int,
                    default=1, help="number of GPUs/CPUs")
parser.add_argument("--accelerator", type=str,
                    default="cpu", help="Device type")
parser.add_argument("--batch_size", type=int,
                    default=512, help="Batch size")
parser.add_argument("--labeled_bs", type=int,
                    default=256, help="Number of labelled samples within batches")
parser.add_argument("--num_epoch", type=int,
                    default=500, help="Number of epoch")
parser.add_argument("--use_consistency", action='store_true',
                    help="Enable consistency loss or no")
parser.add_argument("--dataset", type=str, default='new_adult',
                    help="3 datasets are available: adult, new_adult, acs_employment")
arg = parser.parse_args()

args = {
    "ema_param": 0.99,
    "labeled_bs": arg.labeled_bs,
    "num_epoch": arg.num_epoch,
    "fast_dev_run": 0,
    "devices": arg.devices,
    "accelerator": arg.accelerator,
    "b1": 0.5,
    "b2": 0.999,
    "dataset": arg.dataset,
    "treshold_uncert": 0.3,
    "batch_size": arg.batch_size,
    "consistency_w": arg.use_consistency
}
datasets = {
    'adult': get_old_adult,
    'new_adult': get_adult,
    'compas_race': get_compas_race
}


def train_and_predict(data1, data2, data1_test, consistency_w=args["consistency_w"]):

    data_module = DataModelMissingSensitiveAtt(
        data1=data1, data2=data2, data1_test=data1_test, batch_size=args["batch_size"], val_size=.1, num_workers=4, include_y_in_x=False, labeled_bs=args["labeled_bs"])
    # print(">>>", data_module.n_feature)
    demp = DemographicPredictor(
        input_size=data_module.n_feature, output_size=1, lr=0.001, betas=(args["b1"], args["b2"]), ema_param=args["ema_param"], labelled_bs=args["labeled_bs"], total_epoch=args["num_epoch"], consistency_w=consistency_w, treshold_uncert=args["treshold_uncert"])

    # specify token and project to use NeptuneLogger set it to none
    # neptune_logger = NeptuneLogger(
    # api_token="",
    # project=""
    # )

    neptune_logger = None
    trainer = Trainer(devices=args["devices"],
                      accelerator=args["accelerator"], max_epochs=args["num_epoch"], logger=neptune_logger, fast_dev_run=args["fast_dev_run"])
    trainer.fit(demp, datamodule=data_module)
    trainer.test(demp, datamodule=data_module)
    trainer.logger.log_hyperparams(args)
    # return trainer.predict(demp, dataloaders=data_module.test_dataloader(), return_predictions=True)


data1, data2 = datasets[arg.dataset]()

X, y, s = data1

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, s, test_size=0.3)

results = train_and_predict(data1=(X_train, y_train, s_train), data1_test=(
    X_test, y_test, s_test), data2=data2)

print(results)
