import json
import pickle
import numpy as np
__locations , __data_columns, __model = None, None, None


def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())

    except:
        loc_index = -1
    total_len = len(__data_columns)
    x = np.zeros(total_len)
    x[total_len - 3] = sqft
    x[total_len - 2] = bath
    x[total_len - 1] = bhk

    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0],   2)


def get_location_name():

    return __locations


def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[0:241]
    global  __model
    with open("./artifacts/banglore_home_prices_model.pickle" ,'rb') as f:
        __model = pickle.load(f)

    print("loading saved artifacts...done")


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))
    print(get_estimated_price('Ejipura', 1000, 2, 2))






if __name__ == "__main__":
    print(get_location_name() )