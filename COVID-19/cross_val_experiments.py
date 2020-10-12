import numpy as np
import data
import cross_validation
from sklearn.preprocessing import StandardScaler

h = 28
d = data.Data()
cv = cross_validation.CrossValidate(d, h)
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
    rescaled_data =[]
    for fold, scaler_fold in zip(data, scalers):
        rescaled_fold = []
        for country, scaler in zip(fold, scaler_fold):
           rescaled_fold.append(scaler.inverse_transform(country))
        rescaled_data.append(np.stack(rescaled_fold))
    return rescaled_data

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

#TODO: split, standardire and rescale done.
#TODO: pad and mask models.
