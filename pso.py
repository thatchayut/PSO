#!/usr/bin/python
import pandas as pd
import numpy as np
import random
import math
import process

def main():
    # prepare data
    # default row_to_read = 9357
    row_to_read = 100
    file = pd.read_csv("AirQualityUCI.csv", nrows = row_to_read)
    col_to_read_input = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
    col_to_read_output = ['C6H6(GT)']
    file_input = pd.read_csv("AirQualityUCI.csv", usecols = col_to_read_input)
    file_output = pd.read_csv("AirQualityUCI.csv", usecols = col_to_read_output)
    # num_of_data = file_input.shape[0]
    num_of_data = row_to_read

    # ask for required value
    num_of_folds, num_of_hidden_layers, num_of_nodes_in_hidden_layer = process.getInput(num_of_data)

    # create list of sample's index
    list_sample_index = list(file.index)
    # shuffle list to make it's not affected by order
    random.shuffle(list_sample_index)
    # print(list_sample_index)

    # separate input in to k chunks
    chunk_size = math.ceil(num_of_data / num_of_folds)
    chunk_sample_index = list(process.chunks(list_sample_index, chunk_size))
    num_of_chunks = len(chunk_sample_index)

    print(chunk_sample_index)
    print(len(chunk_sample_index))



if __name__ == '__main__':
    main()