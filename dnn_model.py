# import libraries if available else install them
try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print('One or more libraries are missing. Please install them.')
    print("Run 'pip install -r requirements.txt' in an active terminal in the project directory")
    exit()


class DNNModel:
    def __init__(self) -> None:
        self.SUMMARY_FILE_PATH = '/dnn.png'
        self.learning_rate = 0.001
        self.data = None
        self.model = None
        self.x_data = None
        self.y_data = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # change the learning rate
    def change_learning_rate(self, learning_rate: float) -> None:
        """
            Change the learning rate.

            Args:
                learning_rate (float): The new learning rate.
        """
        self.learning_rate = learning_rate

    # load pickle data
    def load_data(self, data_path: str) -> None:
        """
            Loads the data from the given path.

            Args:
                data_path (str): The path to the pickle file passed as an argument.
        """
        self.data = pd.read_pickle(data_path)

    # sample data
    def sample_data(self, sample_size: int = 12) -> None:
        """
            Sample the data and print the head of the data.

            Args:
                sample_size (int, optional): The size of the sample. Defaults to 12.
        """
        data = self.data.sample(frac=1).reset_index(drop=True)
        # print the head of the data
        print(data.head(sample_size))

    # load dataset
    def load_dataset(self) -> None:
        """
            Loads the dataset from data in the class and create an x_data.
            then displays the head of the data.
        """
        # Load dataset 
        self.y_data = self.data['Target']
        self.x_data = self.data.drop(columns=['Target'])
        # print the head of the data
        print(self.x_data.head())


    # Split the data into training and testing sets
    def split_data(self, test_size: float, random_state: int) -> None:
        """
            Split the data into training and testing sets.

            Args:
                test_size (float): The size of the test set.
                random_state (int): The random state to use for the split.
        """
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x_data, self.y_data,
            test_size=test_size,
            random_state=random_state
        )
        # print the head of the data
        print(f"Train Data head : {self.X_train.head()}")
        print(f"Test Data head : {self.X_test.head()}")
        print(f"Train Data head : {self.y_train.head()}")
        print(f"Test Data head : {self.y_test.head()}")

    # normalize the data
    def normalize_data(self) -> None:
        """
            Normalize the data.
        """
        # normalize the data
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1, input_shape=(1, 9))
        normalizer.adapt(np.array(self.x_data))

    # create model
    def create_model(self) -> None:
        """
            Create a model. and set it to self.model.
        """
        # create a model using the functional api
        _model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.selu),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(64, activation=tf.nn.selu),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(1),
        ])

        self.model = _model

    # compile model
    def compile_model(self) -> None:
        """
            Compile the model.
        """ 
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.losses.MeanAbsoluteError(),  # 'mean_absolute_error',
            metrics=['accuracy']
        )

    # model summary
    def model_summary(self) -> None:
        """
            get the model summary and save it to a file.
            Print the model summary.
        """ 
        summary = self.model.summary()
        tf.keras.utils.plot_model(self.model, to_file=self.SUMMARY_FILE_PATH, show_shapes=True)
        print(summary)

    # train the model
    def train_model(self, epochs: int = 300, batch_size: int = 16) -> None:
        """
            Train the model.

            Args:
                epochs (int, optional): The number of epochs to train the model. Defaults to 300.
                batch_size (int, optional): The batch size to train the model with. Defaults to 16.
        """
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    # plot history
    def plot_loss(self, history: any) -> None:
        """
            take the history and plot the loss.

            Args:
                history (any): the history of the model
        """
        # if no history is passed, use the one in the class
        if history is None:
            history = self.history
        plt.plot(history.history['loss'], label='loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error [Turns]')
        plt.grid(True)

    # evaluate model
    def evaluate_model(self, verbose: int = 0) -> None:
        """
            evaluate the model.

            Args:
                verbose (int, optional): _description_. Defaults to 0.
        """
        self.model.evaluate(self.X_test, self.y_test, verbose=verbose)

    # test data
    def test_data(self, dataset_path:str) -> None:
        """
            Test the model
        """
        test_data = pd.read_csv(dataset_path, index_col=False, header=None)
        # get the data sum
        test_data = test_data.sum()
        # to frames
        test_data = test_data.to_frame()
        test_data = test_data.T
        # print test data
        print(test_data)
