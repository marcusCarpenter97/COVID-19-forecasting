# For each validation fold:
# error scores from evaluate.
# predictions for all countries (destandardized).
# RMSE for all predictions.

class CrossValidate:
    """
    Class for applying rolling cross validation on a tensorflow model trained on time series data.
    """

    def __init__(self, data, horizon):
        """
        data - Data object
        horizon - int
        """
        self.data = data
        self.horizon = horizon
        self.train = []
        self.validate = []
        self.test = []

    def validate():
        pass

    def split_data(self):
        train, val, test_x, test_y = [], [], [], []
        train_size = self.horizon
        # This assumes all countries have the same length.
        # The minus two gives space for the validation and test sets as they will overshoot.
        k_folds = len(self.data.countries[0].data)//self.horizon - 2
        for _ in range(k_folds):
            tr, v, te_x, te_y = self.data.cross_validate(train_size, self.horizon)
            train.append(tr), val.append(v), test_x.append(te_x), test_y.append(te_y)
            train_size += self.horizon
        return train, val, test_x, test_y

# get final shape for last test. (NO?)

