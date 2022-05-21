import pandas as pd


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = []  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        counts.append(row[-1])
    return counts


def load_data():
    """Loading and converting .csv file."""
    data = pd.read_csv("baza.csv")
    conv_data = []
    countries_no = len(data)
    for i in range(countries_no):
        country = [data.Kontynent[i], data.Powierzchnia[i].item(), data.Populacja[i].item(), data.Morze[i], data.Jedzenie[i],
                   data.Alfabet[i], data.Ustroj[i], data.Kraj[i]]
        conv_data.append(country)
    return conv_data
