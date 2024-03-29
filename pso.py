#!/usr/bin/python
import pandas as pd
import numpy as np
import random
import math
import process
import copy
import time

def main():
    start_time = time.time()
    # prepare data
    # default row_to_read = 8029
    # default row_to_read_5_days = 8149
    # default row_to_read_10_days = 8029    
    row_to_read = 8029
    file = pd.read_csv("AirQualityUCI.csv", nrows = row_to_read)
    col_to_read_input = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
    col_to_read_output_5_days = ['Next_5_days_C6H6(GT)']
    col_to_read_output_10_days = ['Next_10_days_C6H6(GT)']
    file_input = pd.read_csv("AirQualityUCI_edited.csv", usecols = col_to_read_input, nrows = row_to_read)
    file_output_5_days = pd.read_csv("AirQualityUCI_edited.csv", usecols = col_to_read_output_5_days, nrows = row_to_read)
    file_output_10_days = pd.read_csv("AirQualityUCI_edited.csv", usecols = col_to_read_output_10_days, nrows = row_to_read)
    num_of_data = row_to_read

    # ask for required value
    num_of_folds, num_of_hidden_layers, num_of_nodes_in_hidden_layer = process.getInput(num_of_data)
    while True:
        num_of_gen = input("Number of generations : ")
        if (num_of_gen.isnumeric() == False):
            print("WARNING : Probability must be numeric.")
        elif (int(num_of_gen) < 0):
            print("WARNING : Number of generation must be positive number")
        else:
            break
    num_of_gen = int(num_of_gen)

    while True:
        learning_rate = input("Positive number used to update velocity : ")
        if (float(learning_rate) <= 0):
            print("WARNING : This must be positive number")
        else:
            break
    learning_rate = float(learning_rate)

    output_file_name = input("Output file name : ")
    output_file = open(str(output_file_name) + ".txt", "w+")

    # create list of sample's index
    list_sample_index = list(file.index)
    # shuffle list to make it's not affected by order
    random.shuffle(list_sample_index)

    # separate input in to k chunks
    chunk_size = math.ceil(num_of_data / num_of_folds)
    chunk_sample = list(process.chunks(list_sample_index, chunk_size))
    num_of_chunks = len(chunk_sample)

    # k-fold cross validation
    for test_sample_index in range(0, num_of_chunks):
        fold_start_time = time.time()
        print("\n------------------------------------------ K : " + str(test_sample_index + 1) + " --------------------------------")
        output_file.write("------------------------------------------ K : " + str(test_sample_index + 1) + " --------------------------------\n")
        chunk_sample_test = chunk_sample[test_sample_index]
        chunk_sample_train = []

        # select training data from all data by excluding testing data
        for train_sample_index in range(0, num_of_chunks):
            if (chunk_sample[train_sample_index] is not chunk_sample_test):
                chunk_sample_train.extend(chunk_sample[train_sample_index])

        # prepare data for training
        file_training_input = file_input.iloc[chunk_sample_train]
        file_training_output_5_days = file_output_5_days.iloc[chunk_sample_train]
        file_training_output_10_days =file_output_10_days.iloc[chunk_sample_train]

        # create list of training data
        num_of_samples = len(chunk_sample_train)
        list_training_input = []
        for row in range(0, num_of_samples):
            list_each_sample = []
            for element in file_training_input.iloc[row, :]:
                list_each_sample.append(element)
            list_training_input.append(list_each_sample)

        list_training_output_5_days = []
        for row in range(0, num_of_samples):
            list_each_sample = []
            for element in file_training_output_5_days.iloc[row, :]:
                list_each_sample.append(element)
            list_training_output_5_days.append(list_each_sample)

        list_training_output_10_days = []
        for row in range(0, num_of_samples):
            list_each_sample = []
            for element in file_training_output_10_days.iloc[row, :]:
                list_each_sample.append(element)
            list_training_output_10_days.append(list_each_sample)
        
        # scaling input and output to be in range (-1, 1)
        list_training_input_normalized = []
        for sample in list_training_input:
            result = process.scaling(sample)
            list_training_input_normalized.append(result)    

        list_training_output_5_days_normalized = []
        for sample_index in range(0, len(list_training_output_5_days)):
            list_total_data_in_sample = []
            list_total_data_in_sample.extend(list_training_input[sample_index])
            list_total_data_in_sample.extend(list_training_output_5_days[sample_index])

            result = process.scaling(list_total_data_in_sample)
            list_training_output_5_days_normalized.append(result[len(result) - 1])

        list_training_output_10_days_normalized = []
        for sample_index in range(0, len(list_training_output_10_days)):
            list_total_data_in_sample = []
            list_total_data_in_sample.extend(list_training_input[sample_index])
            list_total_data_in_sample.extend(list_training_output_10_days[sample_index])

            result = process.scaling(list_total_data_in_sample)
            list_training_output_10_days_normalized.append(result[len(result) - 1])

        # create all particles in this swarm
        particles = {}      
        for i in range(0, num_of_samples):
            key = i
            value = process.createParticle(num_of_hidden_layers, num_of_nodes_in_hidden_layer)
            particles[key] = value

        particles_pbest = {}
        for i in range(0, num_of_samples):
            key = i 
            value = copy.deepcopy(particles[key])     
            particles_pbest[key] = value 
    
        # create a list to record output from each node
        list_all_Y = process.createY(num_of_hidden_layers, num_of_nodes_in_hidden_layer)

        # create a list of pbest (personal best)
        list_pbest = process.createListPbest(num_of_samples)

        # create velocity of each particles
        particles_velocity = {}
        for i in range(0, num_of_samples):
            key = i
            velocity = process.createListVelocity(num_of_hidden_layers, num_of_nodes_in_hidden_layer)
            particles_velocity[key] = velocity

        # TRAINING
        # Iterate through generations
        print(" #### TRAINING ####")
        print("Size fo training data : " + str(len(chunk_sample_train)))
        for count_generation in range(0, num_of_gen):
            print(" #### Generation " + str(count_generation + 1) + " ####")
            list_all_Y = process.createY(num_of_hidden_layers, num_of_nodes_in_hidden_layer)
            # Forwarding
            for i in range(0, num_of_samples):
                # calcualte output for each node in hidden layers
                for layer_index in range(0, num_of_hidden_layers):
                    for node_index in range(0, num_of_nodes_in_hidden_layer[layer_index]):
                        result = 0
                        # weight index is between 1 to len(particles) because weight_index '0' is weight bias
                        num_of_weight = len(particles[i][layer_index][node_index])
                        for weight_index in range(1, num_of_weight):
                            # for node in the 1st hidden layer
                            if (layer_index == 0):
                                # index of list_training_input_normalized must be the same index as the one for an individual
                                for element in list_training_input_normalized[0]:
                                    result += (element * particles[i][layer_index][node_index][weight_index])
                            # for other layers
                            else:
                                # y_this_node = sum(y_previous_node * weight_to_this_node)
                                for element in list_all_Y[layer_index - 1]:
                                    result += (element * particles[i][layer_index][node_index][weight_index])
                        # add bias to result (weight_index '0' is weight bias)
                        result += particles[i][layer_index][node_index][0]
                        # apply activation function to result
                        result = process.sigmoid(result)
                        list_all_Y[layer_index][node_index] = result

                # calculate output for output layer
                num_of_output = 2
                last_hidden_layer_index = len(particles[i]) - 2
                last_layer_index = len(particles[i]) - 1
                for output_index in range(0, num_of_output):
                    # output = sum(y_previous_node * weight_to_this_node)
                    result = 0
                    for weight_index in  range(0, len(particles[i][last_hidden_layer_index][output_index])):
                        for element in list_all_Y[len(list_all_Y) - 1]:
                            result += (element * particles[i][last_hidden_layer_index][output_index][weight_index])
                    # add bias to result (weight_index '0' is weight bias)
                    result += particles[i][last_layer_index][output_index][0]
                    result = process.sigmoid(result)
                    list_all_Y[last_layer_index][output_index] = result
                actual_output = list_all_Y[last_layer_index]
                desired_output_5_days = list_training_output_5_days_normalized[i]
                desired_output_10_days = list_training_output_10_days_normalized[i]

                # calculate fitness of each particle
                fitness_value = process.mae(actual_output, desired_output_5_days, desired_output_10_days)

                # compare the performance of each individual to its best performance (pbest)
                if (fitness_value < list_pbest[i]):
                    # change pbest of this particle
                    list_pbest[i] = fitness_value
                    particles_pbest[i] = copy.deepcopy(particles[i])

                # change the velocity vector of each particles
                for layer_index in range(0, num_of_hidden_layers):
                    for node_index in range(0, num_of_nodes_in_hidden_layer[layer_index]):
                        # weight index is between 1 to len(particles) because weight_index '0' is weight bias
                        num_of_velocity = len(particles_velocity[i][layer_index][node_index])
                        for weight_index in range(0, num_of_velocity):
                            previous_velocity  = particles_velocity[i][layer_index][node_index][weight_index]
                            xpbest = particles_pbest[i][layer_index][node_index][weight_index]
                            x = particles[i][layer_index][node_index][weight_index]
                            velocity = previous_velocity + (learning_rate * (xpbest - x))
                            # update velocity
                            particles_velocity[i][layer_index][node_index][weight_index] = velocity

                            # update weight
                            particles[i][layer_index][node_index][weight_index] = x + particles_velocity[i][layer_index][node_index][weight_index]
                num_of_output = 2
                last_hidden_layer_index = len(particles[i]) - 2
                last_layer_index = len(particles[i]) - 1
                for output_index in range(0, num_of_output):
                    for weight_index in  range(0, len(particles[i][last_hidden_layer_index][output_index])):
                        for element in list_all_Y[len(list_all_Y) - 1]:
                            previous_velocity  = particles_velocity[i][last_layer_index][output_index][weight_index]
                            xpbest = particles_pbest[i][last_layer_index][output_index][weight_index]
                            x = particles[i][last_layer_index][output_index][weight_index]
                            velocity = previous_velocity + (learning_rate * (xpbest - x))
                            # update velocity
                            particles_velocity[i][last_layer_index][output_index][weight_index] = velocity

                            # update weight
                            particles[i][last_layer_index][output_index][weight_index] = x + particles_velocity[i][last_layer_index][output_index][weight_index]

        # Testing
        # prepare testing data
        print()
        print(" #### TESTING ####")
        file_testing_input = file_input.iloc[chunk_sample_test]
        file_testing_output_5_days = file_output_5_days.iloc[chunk_sample_test]
        file_testing_output_10_days = file_output_10_days.iloc[chunk_sample_test]
        # create list testing data
        num_of_samples_to_test = len(chunk_sample_test)
        print("Size of testing data : " + str(num_of_samples_to_test))
        list_testing_input = []
        for row in range(0, num_of_samples_to_test):
            list_each_sample = []
            for element in file_testing_input.iloc[row, :]:
                list_each_sample.append(element)
            list_testing_input.append(list_each_sample)
        
        list_testing_output_5_days = []
        for row in range(0, num_of_samples_to_test):
            list_each_sample = []
            for element in file_testing_output_5_days.iloc[row, :]:
                list_each_sample.append(element)
            list_testing_output_5_days.append(list_each_sample)        

        list_testing_output_10_days = []
        for row in range(0, num_of_samples_to_test):
            list_each_sample = []
            for element in file_testing_output_10_days.iloc[row, :]:
                list_each_sample.append(element)
            list_testing_output_10_days.append(list_each_sample)     

        # scaling input and output to be in range (-1, 1)
        list_testing_input_normalized = []
        for sample in list_testing_input:
            result = process.scaling(sample)
            list_testing_input_normalized.append(result)    

        list_testing_output_5_days_normalized = []
        for sample_index in range(0, len(list_testing_output_5_days)):
            list_total_data_in_sample = []
            list_total_data_in_sample.extend(list_testing_input[sample_index])
            list_total_data_in_sample.extend(list_testing_output_5_days[sample_index])

            result = process.scaling(list_total_data_in_sample)
            list_testing_output_5_days_normalized.append(result[len(result) - 1])

        list_testing_output_10_days_normalized = []
        for sample_index in range(0, len(list_testing_output_10_days)):
            list_total_data_in_sample = []
            list_total_data_in_sample.extend(list_testing_input[sample_index])
            list_total_data_in_sample.extend(list_testing_output_10_days[sample_index])

            result = process.scaling(list_total_data_in_sample)
            list_testing_output_10_days_normalized.append(result[len(result) - 1])

        # create a list to record output from each node
        list_all_Y = process.createY(num_of_hidden_layers, num_of_nodes_in_hidden_layer)

        # Forwarding
        total_error = 0
        for i in range(0, num_of_samples_to_test):
            # calcualte output for each node in hidden layers
            for layer_index in range(0, num_of_hidden_layers):
                for node_index in range(0, num_of_nodes_in_hidden_layer[layer_index]):
                    result = 0
                    # weight index is between 1 to len(particles) because weight_index '0' is weight bias
                    num_of_weight = len(particles_pbest[i][layer_index][node_index])
                    for weight_index in range(1, num_of_weight):
                        # for node in the 1st hidden layer
                        if (layer_index == 0):
                            # index of list_training_input_normalized must be the same index as the one for an individual
                            for element in list_testing_input_normalized[0]:
                                result += (element * particles_pbest[i][layer_index][node_index][weight_index])
                        # for other layers
                        else:
                            # y_this_node = sum(y_previous_node * weight_to_this_node)
                            for element in list_all_Y[layer_index - 1]:
                                result += (element * particles_pbest[i][layer_index][node_index][weight_index])
                    # add bias to result (weight_index '0' is weight bias)
                    result += particles_pbest[i][layer_index][node_index][0]
                    # apply activation function to result
                    result = process.sigmoid(result)
                    list_all_Y[layer_index][node_index] = result

            # calculate output for output layer
            num_of_output = 2
            last_hidden_layer_index = len(particles_pbest[i]) - 2
            last_layer_index = len(particles_pbest[i]) - 1
            for output_index in range(0, num_of_output):
                # output = sum(y_previous_node * weight_to_this_node)
                result = 0
                for weight_index in  range(0, len(particles_pbest[i][last_hidden_layer_index][output_index])):
                    for element in list_all_Y[len(list_all_Y) - 1]:
                        result += (element * particles_pbest[i][last_hidden_layer_index][output_index][weight_index])
                # add bias to result (weight_index '0' is weight bias)
                result += particles_pbest[i][last_layer_index][output_index][0]
                result = process.sigmoid(result)
                list_all_Y[last_layer_index][output_index] = result
            actual_output = list_all_Y[last_layer_index]
            desired_output_5_days = list_testing_output_5_days_normalized[i]
            desired_output_10_days = list_testing_output_10_days_normalized[i]

            print("Testing sample #" + str(i + 1))
            print("Actual Output : " + str(actual_output))
            print("Desired Output (5 days) : " + str(desired_output_5_days))
            print("Desired Output (10 days) : " + str(desired_output_10_days))

            # calculate fitness of each particle
            fitness_value = process.mae(actual_output, desired_output_5_days, desired_output_10_days)
            total_error += fitness_value
        error = (total_error / num_of_samples_to_test)
        error = round(error, 7)
        print()
        print("Mean Average Error of fold-" + str(test_sample_index + 1) + " : " + str(error))
        output_file.write(" Mean Average Error of fold-" + str(test_sample_index + 1) + " : " + str(error) + "\n")
        fold_end_time = time.time()
        fold_elapse_time_second = fold_end_time - fold_start_time
        fold_elapse_time_minute = round((fold_elapse_time_second / 60), 4)
        print("Time elapse in this fold = " + str(fold_elapse_time_minute) +" minutes")
        output_file.write(" Time elapse in this fold = " + str(fold_elapse_time_minute) +" minutes\n")
    end_time = time.time()
    total_elapse_time_second = end_time - start_time
    total_elapse_time_minute = round((total_elapse_time_second / 60), 4)
    print("\n Total elapse time : " + str(total_elapse_time_minute) + " minutes")
    output_file.write("Total elapse time : " + str(total_elapse_time_minute) + " minutes")

if __name__ == '__main__':
    main()