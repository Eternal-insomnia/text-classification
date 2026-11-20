import argparse
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml
from models import define_logreg
from scipy.sparse import load_npz

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(name=__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with open(file="params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(stream=f)

    match params["model"]["name"]:
        case "logreg":
            logger.info(msg="Trainig LogReg model...")
            model = define_logreg(
                c=params["model"]["c"],
                max_iter=params["model"]["max_iter"],
                random_state=params["random_state"],
            )
        case _:
            logger.error(msg="Model not found")
            return

    if model:
        X_train = load_npz(file=args.datadir / "X_train.npz")
        y_train = pd.read_parquet(path=args.datadir / "train.parquet")["label"]

        start_time = time.time()

        model.fit(X=X_train, y=y_train)

        train_time = time.time() - start_time
        logger.info(msg=f"Train time {train_time:.2f}")

        model_path = args.out / f"{params['model']['name']}.joblib"
        args.out.mkdir(parents=True, exist_ok=True)
        joblib.dump(value=model, filename=model_path)
        logger.info(msg=f"Saved model {params['model']['name']} to {model_path}")


if __name__ == "__main__":
    main()
