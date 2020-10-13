import numpy as np
from sklearn.preprocessing import StandardScaler
import data
import cross_validation

PAD_VAL = -10000
H = 28
F = 3
PADDING = np.full((H, F), PAD_VAL)

d = data.Data()
cv = cross_validation.CrossValidate(d, H)
train, val, test_x, test_y = cv.split_data()

def standardize_data(data):
    """
    data - list
    """
    scaled_data = []
    scalers = []
    for fold in data:
        scaled_fold = []
        scalers_fold = []
        for country in fold:
            scaler = StandardScaler()
            scaled_fold.append(scaler.fit_transform(country))
            scalers_fold.append(scaler)
        scaled_data.append(np.stack(scaled_fold))
        scalers.append(scalers_fold)
    return scaled_data, scalers

def destandardize_data(data, scalers):
    """
    data - list
    """
    rescaled_data = []
    for fold, scaler_fold in zip(data, scalers):
        rescaled_fold = []
        for country, scaler in zip(fold, scaler_fold):
            rescaled_fold.append(scaler.inverse_transform(country))
        rescaled_data.append(np.stack(rescaled_fold))
    return rescaled_data

def pad_data(data, offset=0):
    """
    data - list
    """
    padded_scaled_data = []
    folds = len(data) - offset
    for fold in data:
        padded_fold = []
        fold_padding = np.repeat(PADDING, folds).reshape(H*folds, F)
        for row in fold:
            padded_fold.append(np.insert(row, 0, fold_padding, axis=0))
        folds -= 1
        padded_scaled_data.append(np.stack(padded_fold))
    return padded_scaled_data

idx = 1
for tr, v, te_x, te_y in zip(train, val, test_x, test_y):
    print(f"{idx}")
    idx += 1
    print(f"train fold {tr.shape}")
    print(f"val fold {v.shape}")
    print(f"test input fold {te_x.shape}")
    print(f"test output fold {te_y.shape}")

print("Pre scale example")
print(test_y[2][0])

scaled_train, _ = standardize_data(train)
scaled_val, _ = standardize_data(val)
scaled_test_x, _ = standardize_data(test_x)
scaled_test_y, test_y_scalers = standardize_data(test_y)

print(scaled_test_y[2][0])

idx = 1
for tr, v, te_x, te_y in zip(scaled_train, scaled_val, scaled_test_x, scaled_test_y):
    print(f"{idx}")
    idx += 1
    print(f"train fold {tr.shape}")
    print(f"val fold {v.shape}")
    print(f"test input fold {te_x.shape}")
    print(f"test output fold {te_y.shape}")

rescaled_test_y = destandardize_data(scaled_test_y, test_y_scalers)
print(rescaled_test_y[2][0])

padded_scaled_train = pad_data(scaled_train)
padded_scaled_test_x = pad_data(scaled_test_x, offset=1)

print(f"padded scaled train : {len(padded_scaled_train)}")
for fold in padded_scaled_train:
    print(f"padded scaled train : {fold.shape}")

print(f"padded scaled test x: {len(padded_scaled_test_x)}")
for fold in padded_scaled_test_x:
    print(f"padded scaled test x: {fold.shape}")

# make data multi output.
# make input use the names.
# make model use masking.
# train model on each validation fold repreat for each regularizer experiment.
# save results.
