#!/usr/bin/python
import pandas as pd
import math
import process

def main():
    # prepare data
    file = pd.read_csv("AirQualityUCI.csv")
    col_to_read_input = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
    col_to_read_output = ['C6H6(GT)']
    file_input = pd.read_csv("AirQualityUCI.csv", usecols = col_to_read_input)
    file_output = pd.read_csv("AirQualityUCI.csv", usecols = col_to_read_output)
    num_of_data = file_input.shape[0]

    # ask for required data
    while True:
        num_of_folds = input("Number of folds: ")
        if (num_of_folds.isnumeric() == False):
            print("WARNING : Invalid input must be numeric")
        elif(int(num_of_folds) > num_of_data):
            print("WARNING : Number of folds exceeds the size of a dataset")
        elif(int(num_of_folds) <= 1):
            print("WARNING : Number of folds cannot lower than or equal to 1")
        else:
            break
    num_of_folds = int(num_of_folds)

    # separate input in to k chunks
    chunk_size = math.ceil(num_of_data / num_of_folds)
    chunk_sample_input = list(process.chunks(file_input, chunk_size))
    chunk_sample_output = list(process.chunks(file_output, chunk_size))
    num_of_chunks = len(chunk_sample)



if __name__ == '__main__':
    main()