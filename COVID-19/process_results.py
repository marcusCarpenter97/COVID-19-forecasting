import os
import csv
import data

TARGET_DIR = "cross_val_results"

def save_country_names():
    d = data.Data()
    names = [country.name for country in d.countries]
    path = os.path.join(TARGET_DIR, "country_names.csv")
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(names)

def load_country_names():
    path = os.path.join(TARGET_DIR, "country_names.csv")
    with open(path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        results = [row for row in reader]
    return results[0]

if __name__ == "__main__":

    save_country_names()
