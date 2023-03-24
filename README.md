# inter-turn
from models import cnn_model as cnn
from models import dnn_model as dnn
from models import fault_simulation_model as flt
import tensorflow as tf
import numpy as np
os.chdir("C:\\Users\\Edwin\\Desktop\\PROJECT\\AI")
def main():
    '''
    cnn_data_file = "C:\\Users\\Edwin\\Desktop\\PROJECT\\AI\\dataset\\finaldataset.npz" # cnn data file path

    # create an instance of the cnn model
    cnn_model_instance = cnn.CNNModel()
    """ you can change the model version before anything else """
    # cnn_model_instance.change_version(2) # comment or uncomment this line to change the model version
    cnn_model_instance.load_data(cnn_data_file, mmap_mode='r')
    # process the data
    cnn_model_instance.process_data(cnn_model_instance.dataset)
    # create a model
    cnn_model_instance.create_model()
    '''
    '''
        before compiling, you can change batch size, epochs, and learning rate for the optimizer
    '''
    '''
    # # change epochs
    # cnn_model_instance.change_epochs(10) # comment or uncomment this line to change epochs
    # # change batch size
    # cnn_model_instance.change_batch_size(32) # comment or uncomment this line to change batch size
    # # change learning rate
    # cnn_model_instance.change_learning_rate(0.001) # comment or uncomment this line to change learning rate


    # compile model
    cnn_model_instance.compile_model()
    # train the model 
    cnn_model_instance.train_model()
    # save the model
    cnn_model_instance.save_model()
    # save weights
    cnn_model_instance.save_weights()
    # visualize the model
    cnn_model_instance.visualize_model()

'''

    
    #dnn_data_file = "C:\\Users\\Edwin\\Desktop\\PROJECT\\AI\\dataset\\percentShortsummed.pkl" # dnn data file path
   # data_set_path_csv = "C:\\Users\\Edwin\\Desktop\\PROJECT\\AI\\dataset\\percent_csvs\\0.3.csv" # data set path for the csv file

    # create an instance of the dnn model
    dnn_model_instance = dnn.DNNModel()
  
    # load the data
    #dnn_model_instance.load_data(data_path=dnn_data_file)
    # sample the data
    #dnn_model_instance.sample_data(sample_size=10)
    # load dataset
    #dnn_model_instance.load_dataset()
    # split the data
   # dnn_model_instance.split_data(test_size=0.2, random_state=42)
    
    # normalize the data
    dnn_model_instance.normalize_data()
    # create a model
    dnn_model_instance.create_model()
    """ 
        you can change the learning rate for the optimizer
    """
    
    # change the learning rate
    # dnn_model_instance.change_learning_rate(0.001) # comment or uncomment this line to change learning rate
    # compile the model
    dnn_model_instance.compile_model()
    # train the model
    dnn_model_instance.train_model()
    # model summary
    dnn_model_instance.model_summary()
    # evaluate the model
    dnn_model_instance.evaluate_model()
    # plot the loss
    dnn_model_instance.plot_loss()
    # test the data
    #dnn_model_instance.test_data(dataset_path=data_set_path_csv)

    '''
    # create an instance of the fault simulation model
    fault_simulation_model_instance = flt.FaultSimulation()

    # percent model path
    model_path = 'C:\\Users\\Edwin\\Desktop\\PROJECT\\NN_EDWIN\\src\\data\\Faulty\\Scalogram\\Phase A\\Primary\\turns_0.2\\sec_2\\time_0.2'
    # dataset path
    data_set_path = 'energy.csv'
    # images path
    images_path = '.\\data\\images'
    # decompose csv file path
    decompose_csv_path = '.\\data\\decompose.csv'

    # load model
    fault_simulation_model_instance.load_model(
        model_path=model_path,
        # version=4 # uncomment this to change the default version
    )
    # test the model
    fault_simulation_model_instance.test(
        csv_file_path=decompose_csv_path, 
        bulk_file_path=data_set_path,  
        single_file_path=images_path
    )
'''

# run if this is the main file
if __name__ == '__main__':
    main()
