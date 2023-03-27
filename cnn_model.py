# import libraries if available else install them
try:
    # all imports
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import gc
    import matplotlib
    import numpy as np
    import tensorflow_datasets as tfds
    import pickle
   # import visualkeras
    from tensorflow.keras.callbacks import ModelCheckpoint
except ImportError:
    # prompt the user
    print('One or more libraries are missing. Please install them.')
    print("Run 'pip install -r requirements.txt' in an active terminal in the project directory")
    exit()


class CNNModel:
    def __init__(self) -> None:
        matplotlib.use("Agg")
        plt.style.use("ggplot")
        self.VERSION = 1
        self.EPOCHS = 3 # can be changed to increase the number of epochs
        self.LEARNING_RATE = 3e-4
        self.BATCH_SIZE = 32 # can be changed to increase the batch size
        self.DECAY = self.LEARNING_RATE / self.EPOCHS
        self.dataset = None
        self.model = None
        self.History = None
        self.healthLB = None
        self.phaseLB = None
        self.sideLB = None
        self.test_size = 0.2
        self.random_state = 42
        self.trainX = None
        self.testX = None
        self.trainHealthY = None
        self.testHealthY = None
        self.trainPhaseY = None
        self.testPhaseY = None
        self.trainSideY = None
        self.testSideY = None
        self.health_labels = None 
        self.phase_labels = None
        self.side_labels = None
        self.number= None
        self.side_weightpath = "./v{}_side_weights.hdf5".format(self.VERSION)
        self.phase_weightpath= "./v{}_phase_weights.hdf5".format(self.VERSION)
        
        
        self.optimizer = keras.optimizers.Adam(learning_rate=self.LEARNING_RATE, decay=self.DECAY)
        self.losses = {
            "h_output": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            "p_output": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            "s_output": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        }
        self.metrics = {
            "h_output": "accuracy",
            "p_output": "accuracy",
            "s_output": "accuracy",
        }
        self.lossNames = [
            "loss", 
            "h_output_loss", 
            "p_output_loss", 
            "s_output_loss"
        ]
        self.accuracyNames = [
            "h_output_accuracy", 
            "p_output_accuracy", 
            "s_output_accuracy"
        ]

    # collect gc
    @staticmethod
    def collect_garbage() -> any:
        gc.collect()

    # change the version of the model
    def change_version(self, version: int) -> None:
        """
            Change the version of the model

            Args:
                version (int): The version of the model
        """
        self.VERSION = version
    
    def monitor(self):
        self.side_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        self.side_weightpath,
                        monitor='val_s_output_accuracy',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        mode='max'
                        )
        self.phase_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        self.phase_weightpath,
                        monitor='val_p_output_accuracy',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        mode='max')
        

    # change epochs
    def change_epochs(self, epochs: int) -> None:
        """
            Change the number of epochs at which the model is trained

            Args:
                epochs (int): The number of epochs at which the model is trained
        """
        self.EPOCHS = epochs


    # change the batch size
    def change_batch_size(self, batch_size: int) -> None:
        """
            Change the batch size at which the model is trained

            Args:
                batch_size (int): The batch size at which the model is trained
        """
        self.BATCH_SIZE = batch_size

    # change the learning rate
    def change_learning_rate(self, learning_rate: float) -> None:
        """
            Change the learning rate at which the model is trained
            once this value is changes, the learning rate will be updated
            and the decay and optimizer will be updated accordingly

            Args:
                learning_rate (float): The learning rate at which the model is trained
        """
        self.LEARNING_RATE = learning_rate
        self.DECAY = self.LEARNING_RATE / self.EPOCHS
        self.optimizer = keras.optimizers.Adam(learning_rate=self.LEARNING_RATE, decay=self.DECAY)

    

    # load data
    def load_data(self, file_url: str, mmap_mode: str) -> None:
        """
            Load the dataset to train and test the model on

            Args:
                file_url (str): The path to the file to be loaded
                mmap_mode (str): the mode in which the file is to be loaded
        """
        self.dataset = np.load(file_url, mmap_mode=mmap_mode)
        # collect any garbage
        self.collect_garbage()
        

    # save data
    def save_data(self, file_url: str, mode: str, data: any) -> None:
        """
            This saves the data to a file in pickle format

            Args:
                file_url (str): The path to the file to be saved
                mode (str): the mode in which the file is to be saved
                data (any): the data to be saved
        """
        file = open(file_url, mode)
        file.write(pickle.dumps(data))
        # close the file after writing
        file.close()
        # delete the file from memory
        del file
        self.collect_garbage()

    
    # process data
    def process_data(self, dataset: any) -> None:
        """
            Process the data
            Args:
                dataset (any): The dataset to be processed
        """
        data = dataset['data']
        self.health_labels = dataset['healthLabels']
        self.phase_labels = dataset['phaseLabels']
        self.side_labels = dataset['sideLabels']
        self.percent_labels = dataset['percentLabels']

        # Encode the labels 
        self.healthLB = LabelEncoder()
        self.phaseLB = LabelEncoder()
        self.sideLB = LabelEncoder()
        self.percentLB = LabelEncoder()

        # Preprocess the data
        self.health_labels = self.healthLB.fit_transform(self.health_labels)
        self.phase_labels = self.phaseLB.fit_transform(self.phase_labels)
        self.side_labels = self.sideLB.fit_transform(self.side_labels)
        self.percent_labels = self.percentLB.fit_transform(self.percent_labels)

        # Split dataset into train-test 
        split = self.split_data(
            data           
        )
        self.collect_garbage()
        (
            self.trainX,
            self.testX,
            self.trainHealthY,
            self.testHealthY,
            self.trainPhaseY,
            self.testPhaseY,
            self.trainSideY,
            self.testSideY,
        ) = split
        # Save binaries to disk for future recall
        self.save_data(file_url='./healthLB.pickle', mode='wb', data=self.healthLB)
        self.save_data(file_url='./phaseLB.pickle', mode='wb', data=self.phaseLB)
        self.save_data(file_url='./sideLB.pickle', mode='wb', data=self.sideLB)
        self.save_data(file_url='./percentLB.pickle', mode='wb', data=self.percentLB)
    # split data
    def split_data(self, data: any) -> any:
        """
            This method is called from the process data method to split the data into training and testing data

            Args:
                data (any): takes in the dataset to be processed

            Returns:
                any: Returns the split data after splitting it
        """
        # Split dataset into train-test 
        split = train_test_split(
            data,
            self.health_labels,
            self.phase_labels,
            self.side_labels,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return split



    def create_model(self) -> None:
        """
            Create the model

            Returns:
                None: this creates the model and adds it to the self.model variable
        """
        inputs = tf.keras.layers.Input(shape=(256, 256, 18), name='input_layer')
        lambda_layer = tf.keras.layers.Lambda(lambda value: value / 255)(inputs)
        xp = tf.keras.layers.Conv2D(72, 3, padding='same', activation=tf.nn.relu)(lambda_layer)
        xp = tf.keras.layers.Conv2D(72, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.MaxPooling2D()(xp)  # by default uses 2,2
        xp = tf.keras.layers.BatchNormalization()(xp)
        xp = tf.keras.layers.Conv2D(96, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.Conv2D(96, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.MaxPooling2D()(xp)
        xp = tf.keras.layers.BatchNormalization()(xp)
        xp = tf.keras.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.MaxPooling2D()(xp)
        xp = tf.keras.layers.BatchNormalization()(xp)
        xp = tf.keras.layers.Conv2D(176, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.Conv2D(176, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.MaxPooling2D()(xp)
        xp = tf.keras.layers.BatchNormalization()(xp)
        xp = tf.keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.MaxPooling2D()(xp)
        xp = tf.keras.layers.BatchNormalization()(xp)
        xp = tf.keras.layers.Conv2D(384, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.Conv2D(384, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.MaxPooling2D()(xp)
        xp = tf.keras.layers.BatchNormalization()(xp)
        xp = tf.keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.MaxPooling2D()(xp)
        xp = tf.keras.layers.BatchNormalization()(xp)
        xp = tf.keras.layers.Dropout(0.25)(xp)
        xp = tf.keras.layers.Flatten()(xp)
        xp = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.BatchNormalization()(xp)
        xp = tf.keras.layers.Dropout(0.25)(xp)
        xp = tf.keras.layers.Dense(512, activation=tf.nn.relu)(xp)
        xp = tf.keras.layers.Dropout(0.25)(xp)
        health_out = tf.keras.layers.Dense(
            len(self.healthLB.classes_),
            activation=tf.nn.softmax,
            name='h_output'
        )(xp)
        phase_out = tf.keras.layers.Dense(
            len(self.phaseLB.classes_),
            activation=tf.nn.softmax,
            name='p_output'
        )(xp)
        side_out = tf.keras.layers.Dense(
            len(self.sideLB.classes_),
            activation=tf.nn.softmax,
            name='s_output'
        )(xp)
        model = tf.keras.models.Model(
            inputs=inputs,
            outputs=[health_out, phase_out, side_out],
            name=f'Detector_version_{self.VERSION}',
        )
        self.model = model


    # model summary
    def model_summary(self, model: any) -> None:
        """
            Prints the model summary

            Args:
                model (any): this takes in the model after training
                it takes the model after training if no model is passed through
        """
        # if no model is passed in, use the model that was trained
        if model is None:
            model = self.model
        summary_file_path = f'./detector_v{self.VERSION}.png'
        model.summary()
        tf.keras.utils.plot_model(model, to_file=summary_file_path, show_shapes=True)
        visualkeras.layered_view(model,to_file= 'modeldrawing.png', legend=True, draw_volume=True)

    # compile model
    def compile_model(self) -> None:
        """
            Compile the model
        """
        self.model.compile(optimizer=self.optimizer, loss=self.losses, metrics=self.metrics)


    # train model
    def train_model(self) -> None:
        """
            Train the model
        """
        self.monitor()
        self.History = self.model.fit(
            x=self.trainX,
            y={
                "h_output": self.trainHealthY,
                "p_output": self.trainPhaseY,
                "s_output": self.trainSideY,
            },
            validation_data=(
                self.testX,
                {
                    "h_output": self.testHealthY,
                    "p_output": self.testPhaseY,
                    "s_output": self.testSideY,
                }
            ),
            epochs=self.EPOCHS,
            shuffle=True,
            callbacks=[self.phase_checkpoint, self.side_checkpoint]
        )

    def print_image(self) -> None:
        self.number=1
        print(self.trainX[self.number])

    def see_image(self) -> None:
        self.number=1
        ff = tfds.visualization.show_examples(self.dataset,tfds.load('mnist',with_info=True))
        


    # save model to disk
    def save_model(self) -> None:
        self.model.save(f'./detector_v{self.VERSION}.h5', save_format='h5')

    def load_model(self) -> None:
        self.model.load_

    # save weight
    def save_weights(self) -> None:
        self.model.save_weights(f'./v{self.VERSION}_model_weights.hdf5')

    def load_weights(self) -> None:
        self.model.load_weights(f'./v{self.VERSION}_model_weights.hdf5')

    # plot accuracy
    def visualize_model(self) -> None:
        (fig, ax) = plt.subplots(5, 1, figsize=(13, 13))
        # loop over the loss names 
        for (index, loss) in enumerate(self.lossNames):
            # plot the loss for both the training and validation data 
            title = "Loss for {}".format(loss) if loss != "loss" else "Total loss"
            ax[index].set_title(title)
            ax[index].set_xlabel("Epoch #")
            ax[index].set_ylabel("Loss")
            ax[index].plot(np.arange(0, self.EPOCHS), self.History.history[loss], label=loss)
            ax[index].plot(np.arange(0, self.EPOCHS), self.History.history["val_" + loss], label="val_" + loss)
            ax[index].legend()
            # save the losses figure 
            plt.tight_layout()
            plt.savefig(f"./v_{self.VERSION}_ls.png")
            plt.close()

        (fig, ax) = plt.subplots(4, 1, figsize=(8, 8))
        # loop over the accuracy names 
        for (index, loss) in enumerate(self.accuracyNames):
            # plot the loss for both the training and validation data	
            ax[index].set_title(f"Accuracy for {loss}")
            ax[index].set_xlabel("Epoch #")
            ax[index].set_ylabel("Accuracy")
            ax[index].plot(np.arange(0, self.EPOCHS), self.History.history[loss], label=loss)
            ax[index].plot(np.arange(0, self.EPOCHS), self.History.history["val_" + loss], label="val_" + loss)
            ax[index].legend()
            # save the accuracy figure 
            plt.tight_layout()
            plt.savefig(f"./v_{self.VERSION}_acs.png")
            plt.close()
