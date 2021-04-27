import logging
import datetime
from time import time
import data
import experiment

if __name__ == "__main__":
    logging.basicConfig(filename="logger.log", format="%(asctime)s %(name)s %(levelname)s: %(message)s", level=logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)

    dat = data.Data()
    dat.prepare_data()

    logging.info(f"Number of days in data: {len(dat.countries[0].data)}")
    logging.info(f"Current date and time: {datetime.datetime.now()}")

    exp = experiment.Experiment(dat.val_scalers, dat.test_y_scalers)
    start = time()
    exp.run_experiments(dat.horizon, dat.pad_val, dat.padded_scaled_train, dat.padded_scaled_test_x, dat.multi_out_scaled_val,
                        dat.multi_out_scaled_test_y, dat.val, dat.test_y, dat.encoded_names)
    end = time()
    logging.info(f"Time taken to run all models: {end - start} seconds")
