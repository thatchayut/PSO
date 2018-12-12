#!/usr/bin/python
import pandas as pd
import numpy as np
import random
import math
import process

def main():
    # prepare data
    # default row_to_read = 9357
    # default row_to_read_5_days = 9237
    # default row_to_read_10_days = 9117    
    # row_to_read_5_days = 9237
    # row_to_read_10_days = 9117
    row_to_read = 9117
    file = pd.read_csv("AirQualityUCI.csv", nrows = row_to_read)
    col_to_read_input = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
    col_to_read_output_5_days = ['Next_5_days_C6H6(GT)']
    col_to_read_output_10_days = ['Next_10_days_C6H6(GT)']
    file_input = pd.read_csv("AirQualityUCI_edited.csv", usecols = col_to_read_input, nrows = row_to_read)
    file_output_5_days = pd.read_csv("AirQualityUCI_edited.csv", usecols = col_to_read_output_5_days, nrows = row_to_read)
    file_output_10_days = pd.read_csv("AirQualityUCI_edited.csv", usecols = col_to_read_output_10_days, nrows = row_to_read)
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
    chunk_sample = list(process.chunks(list_sample_index, chunk_size))
    num_of_chunks = len(chunk_sample)
    # print(chunk_sample_index)
    # print(len(chunk_sample_index))

    # k-fold cross validation
    for test_sample_index in range(0, num_of_chunks):
        print("\n------------------------------------------ K : " + str(test_sample_index + 1) + " --------------------------------")
        chunk_sample_test = chunk_sample[test_sample_index]
        chunk_sample_train = []

        # select training data from all data by excluding testing data
        for train_sample_index in range(0, num_of_chunks):
            if (chunk_sample[train_sample_index] is not chunk_sample_test):
                chunk_sample_train.extend(chunk_sample[train_sample_index])
        print("Size fo training data : " + str(len(chunk_sample_train)))

        # prepare data for training
        file_training_input = file_input.iloc[chunk_sample_train]
        file_training_output_5_days = file_output_5_days.iloc[chunk_sample_train]
        file_training_output_10_days =file_output_10_days.iloc[chunk_sample_train]
        print(len(file_output_10_days))
        # print(file_training_input)


if __name__ == '__main__':
    main()