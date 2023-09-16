#Import all the libraries I need
try:
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import r2_score
    import gc
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    #import tensorflow_datasets as tfds
    import pickle
    from tensorflow.keras.callbacks import ModelCheckpoint
    import cv2 as cv
    from imutils import paths
    import argparse
    import pandas as pd
    import sklearn.metrics
    from sklearn.metrics import confusion_matrix
    import io
    import cv2
    import seaborn as sns
    import itertools
    import os
except ImportError:
    print('One or more libraries are missing. Please install them.')
    print("Run 'pip install -r requirements.txt' in an active terminal in the project directory")
    exit()
os.chdir("C:\\Users\\Edwin\\Desktop\\Career\\Project\\newtry\\")
class CNNModel:
        def __init__(self) -> None:
            matplotlib.use('agg')
            plt.style.use('ggplot')
            self.version = 1
            self.epochs = 20
            self.learning_rate = 1e-6
            self.weight_decay = 5e-4
            self.dataset = None
            self.model = None
            self.batchsize = 32
            self.History = None
            self.healthLB = None
            self.phaseLB = None
            self.sideLB = None
            self.test_size = 0.3
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
            self.side_weightpath = "./v{}_side_weights.hdf5".format(self.version)
            self.phase_weightpath= "./v{}_phase_weights.hdf5".format(self.version)

            self.optimizer = keras.optimizers.Adam(learning_rate = self.learning_rate, decay = self.weight_decay)
            self.losses = { 
                "h_output": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                "p_output": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                "s_output": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                "percent_output": keras.losses.MeanSquaredError()
            }
            self.metrics = {"h_output": "accuracy","p_output": "accuracy", "s_output": "accuracy","percent_output": "mse"}
            self.lossNames = ["loss", "h_output_loss", "p_output_loss", "s_output_loss","percent_output_loss"]
            self.accuracyNames = ["h_output_accuracy", "p_output_accuracy", "s_output_accuracy"]
     
        @staticmethod
        def collect_garbage() -> any:
            gc.collect()
        
        def change_version(self,version :int):
            self.version = version
        
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
            
        def set_epochs(self, epochs: int) -> None:
            self.epochs = epochs

        def set_batchsize(self, batchsize:int):
            self.batchsize = batchsize
        
        def set_learning_rate(self, learning_rate: float) -> None:
            self.learning_rate = learning_rate
            self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, decay=self.weight_decay)
          
        def load_data(self, file_url: str, mmap_mode: str) -> None:
            self.dataset = np.load(file_url, mmap_mode=mmap_mode)
            self.collect_garbage()

        def save_data(self, file_url: str, mode: str, data: any) -> None:
            file = open(file_url, mode)
            file.write(pickle.dumps(data))
            # close the file after writing
            file.close()
            # delete the file from memory
            del file
            self.collect_garbage()

        def process_data(self, dataset: any) -> None:
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
            split = self.split_data(data           
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
                self.trainPercent,
                self.testPercent,
            ) = split
            # Save binaries to disk for future recall
            self.save_data(file_url='./healthLB.pickle', mode='wb', data=self.healthLB)
            self.save_data(file_url='./phaseLB.pickle', mode='wb', data=self.phaseLB)
            self.save_data(file_url='./sideLB.pickle', mode='wb', data=self.sideLB)
            self.save_data(file_url='./percentLB.pickle', mode='wb', data=self.percentLB)

        def split_data(self, data: any) -> any:
            split = train_test_split(
                data,
                self.health_labels,
                self.phase_labels,
                self.side_labels,
                self.percent_labels,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            return split
        
        def create_model(self) -> None:
           

            inputs = tf.keras.layers.Input(shape=(256, 256, 3), name='input_layer')
            x = tf.keras.layers.Lambda(lambda value: value/255)(inputs)
            x = tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(384, 3, padding='same', activation=tf.nn.relu)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
            x = tf.keras.layers.Dropout(0.25)(x)
            health_out = tf.keras.layers.Dense(
            len(self.healthLB.classes_),
            activation=tf.nn.softmax,
            name='h_output'
            )(x)
            phase_out = tf.keras.layers.Dense(
            len(self.phaseLB.classes_),
            activation=tf.nn.softmax,
            name='p_output'
            )(x)
            side_out = tf.keras.layers.Dense(
            len(self.sideLB.classes_),
            activation=tf.nn.softmax,
            name='s_output'
            )(x)
            percent_out = tf.keras.layers.Dense(1, activation = 'linear',name = 'percent_output')(x)
            model = tf.keras.models.Model(
            inputs=inputs,
            outputs=[health_out, phase_out, side_out,percent_out],
            name=f'Detector_version_{self.version}',
            )
            self.model = model

        def compile_model(self) -> None:
            self.model.compile(optimizer=self.optimizer, loss=self.losses, metrics=self.metrics)
            

        def train_model(self) -> None:
            self.monitor()
            history = self.model.fit(
                x=self.trainX,
                y={"h_output": self.trainHealthY, "p_output": self.trainPhaseY,"s_output": self.trainSideY,"percent_output":self.trainPercent},
                validation_data=(self.testX,{"h_output": self.testHealthY,"p_output": self.testPhaseY,"s_output": self.testSideY, "percent_output": self.testPercent}),
                epochs=self.epochs, shuffle=True, callbacks=[self.phase_checkpoint, self.side_checkpoint])
            self.plot_history(history= history)
            
        def plot_history(self, history: any):    
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

            # Health train vs test accuracy
            axs[0, 0].plot(history.history['h_output_accuracy'], label='train')
            axs[0, 0].plot(history.history['val_h_output_accuracy'], label='test')
            axs[0, 0].set_title('Health Accuracy')
            axs[0, 0].legend()

            # Phase train vs test accuracy
            axs[0, 1].plot(history.history['p_output_accuracy'], label='train')
            axs[0, 1].plot(history.history['val_p_output_accuracy'], label='test')
            axs[0, 1].set_title('Phase Accuracy')
            axs[0, 1].legend()

            # Side train vs test accuracy
            axs[1, 0].plot(history.history['s_output_accuracy'], label='train')
            axs[1, 0].plot(history.history['val_s_output_accuracy'], label='test')
            axs[1, 0].set_title('Side Accuracy')
            axs[1, 0].legend()

            # Percent train vs test loss
            axs[1, 1].plot(history.history['percent_output_loss'], label='train')
            axs[1, 1].plot(history.history['val_percent_output_loss'], label='test')
            axs[1, 1].set_title('Percent Loss')
            axs[1, 1].legend()
            plt.tight_layout()
            plt.savefig('training_graphs.png')
            
        def save_model(self) -> None:
            self.model.save(f'./detector_v{self.version}.h5', save_format='h5')

        def save_weights(self) -> None:
            self.model.save_weights(f'./v{self.version}_model_weights.hdf5')

        def load_weights(self) -> None:
            self.model.load_weights(f'./v{self.version}_model_weights.hdf5')

        def visualize_model(self) -> None:
            model_img_file = 'model.png'
            tf.keras.utils.plot_model(self.model, to_file=model_img_file, show_shapes=True, show_layer_activations=True, show_dtype=True, show_layer_names=True)


class FaultSimulation:
    def __init__(self) -> None:
        self.version = 1
        self.model = None
        self.percentModel = None
        self.healthLE = LabelEncoder()
        self.phaseLE = LabelEncoder()
        self.sideLE = LabelEncoder()
        self.health_labels = None
        self.phase_labels = None
        self.side_labels = None
        self.folders = ['\\input_current', '\\output_current']
        self.health_label = None
        self.phase_label = None
        self.side_label = None

        app = argparse.ArgumentParser()
        app.add_argument(
            "-v",
            "--version",
            required=False, 
            default=self.version, 
            help="Model version"
        )
        
        app.add_argument(
            "-b", "--bulk", 
            required=False, 
            default=True,
            type=bool,
            help="Run classification on batch or single"
        )
        args = app.parse_args()

        self.VERSION = args.version
        self.BULK = args.bulk

        print(f"Model Version: {self.version}")
        print(f"Running classification on bulk: {self.BULK is True}")
    
        self.model = keras.models.load_model("C:\\Users\\Edwin\\Desktop\\Career\\Research\\Transformer Inter-turn\\newtry\\detector_v1.h5")
        self.healthLB = pickle.loads(open("C:\\Users\\Edwin\\Desktop\\Career\\Research\\Transformer Inter-turn\\newtry\\healthLB.pickle", "rb").read())
        self.phaseLB = pickle.loads(open("C:\\Users\\Edwin\\Desktop\\Career\\Research\\Transformer Inter-turn\\newtry\\phaseLB.pickle", "rb").read())
        self.sideLB = pickle.loads(open("C:\\Users\\Edwin\\Desktop\\Career\\Research\\Transformer Inter-turn\\newtry\\sideLB.pickle", "rb").read())
        self.percentLB = pickle.loads(open("C:\\Users\\Edwin\\Desktop\\Career\\Research\\Transformer Inter-turn\\newtry\\percentLB.pickle", "rb").read())

    @staticmethod
    def collect_garbage() -> None:
        gc.collect()

    def plot_confusion_matrix(self,cm, class_names, filename):
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        plt.savefig(filename)
        return figure
    
    def plot_r2score(self, y : any, y_pred: any):
        r_squared = r2_score(y,y_pred)
        print(r_squared)
        plt.scatter(y,y_pred)
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        z = np.polyfit(y.ravel(), y_pred.ravel(), 1)
        p = np.poly1d(z)
        plt.plot(y.ravel(), p(y.ravel()), color='red')
        plt.title(f"R-squared = {r_squared}")
        plt.savefig("Percentage Correlation")
        


    def classify_on_bulk(self, file_path: str, mmap_mode: str = 'r'):
        dataset = np.load(file_path, mmap_mode=mmap_mode)
        self.collect_garbage()

        data = dataset['data']
        self.health_labels = dataset['healthLabels']
        self.phase_labels = dataset['phaseLabels']
        self.side_labels = dataset['sideLabels']
        self.percent_labels = dataset['percentLabels']

        # Clear the dataset variable from memory	
        del dataset
        self.collect_garbage()

        self.healthLE = LabelEncoder()
        self.phaseLE = LabelEncoder()
        self.sideLE = LabelEncoder()
        self.percentLE = LabelEncoder()

        health_labels = self.healthLE.fit_transform(self.health_labels)
        phase_labels = self.phaseLE.fit_transform(self.phase_labels)
        side_labels = self.sideLE.fit_transform(self.side_labels)
        percent_labels = self.percentLE.fit_transform(self.percent_labels)

        test_pred_raw = self.model.predict(data)
        test_pred_health = np.argmax(test_pred_raw[0], axis=1)
        test_pred_phase = np.argmax(test_pred_raw[1], axis=1)
        test_pred_side = np.argmax(test_pred_raw[2], axis=1)
        test_pred_percent = test_pred_raw[3]

        
        self.plot_r2score(percent_labels,test_pred_percent)
        self.collect_garbage()
        #results = self.model.evaluate(x=data,y={"h_output": health_labels,"p_output": phase_labels,"s_output": side_labels,"percent_output": percent_labels})
        


    
        
        del data
        self.collect_garbage()
        
        
        # Calculate the confusion matrix.	
        health_cm = sklearn.metrics.confusion_matrix(health_labels, test_pred_health)
        phase_cm = sklearn.metrics.confusion_matrix(phase_labels, test_pred_phase)
        side_cm = sklearn.metrics.confusion_matrix(side_labels, test_pred_side)
        
        # delete parameters
        del health_labels
        del phase_labels
        del side_labels
        del percent_labels
        del test_pred_raw
        del test_pred_health
        del test_pred_phase
        del test_pred_side
        del test_pred_percent
        self.collect_garbage()
        
        
        # Log the confusion matrix as an image summary. 
        self.plot_confusion_matrix(
            health_cm,
            self.healthLB.classes_,
            filename=f'results/HealthConfusion.png'
        )
        self.plot_confusion_matrix(
            phase_cm,
            class_names=self.phaseLB.classes_,
            filename=f'results/PhaseConfusion.png'
        )
        self.plot_confusion_matrix(
            side_cm,
            class_names=self.sideLB.classes_,
            filename=f'results/SideConfusion.png',
        )     
        print('[DONE]: Finished classifying test images')


    def classify_on_single(self, file_path: str):
        vconcate = []
        self.folders = ['input_current','output_current']
        for folder in self.folders:
            self.filepath = file_path + folder
            newlist = []
            for imagePath in sorted(list(paths.list_images(self.filepath))):        
                    image = cv2.imread(imagePath)
                    image = cv2.resize(image, [256,256])
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                               
                    newlist.append(image)                                                  
            hconcat_image = np.concatenate(newlist,axis = 1)
            vconcate.append(hconcat_image)                        
        vconcat_image = np.concatenate(vconcate,axis=0)
        image = cv2.resize(vconcat_image, [256,256])
        cv2.imwrite('healthytaste.jpg',image)
        cv2.imshow('wow',image)
        cv2.waitKey(0)
        image = np.expand_dims(image,axis = 0)
        result = self.model.predict(image)
        (health, phase, side,percent) = result      
        health_idx = health[0].argmax()
        phase_idx = phase[0].argmax()
        side_idx = side[0].argmax()
        percent_idx= percent[0]
        self.health_label = self.healthLB.classes_[health_idx]
        self.phase_label = self.phaseLB.classes_[phase_idx]
        self.side_label = self.sideLB.classes_[side_idx]
        percent_test =  percent_idx.flatten()
        percent_test = float(percent_test) * 2
        print(percent_idx)
        print(" State:\t\t{}".format(self.health_label))
        if self.health_label == "Faulty":
            print(" Phase:\t\t{}".format(self.phase_label))
            print(" Side:\t\t{}".format(self.side_label))
            print(" Turns:\t\t{}%".format(round(float(percent_test))))



class main():
    cnn_data_file = "C:\\Users\\Edwin\\Desktop\\Career\\Project\\fault_tester.npz"
    '''
    cnn = CNNModel()
    cnn.load_data(cnn_data_file, 'r')
    cnn.process_data(cnn.dataset)
    cnn.create_model()
    cnn.visualize_model()
    cnn.set_batchsize(32)
    cnn.set_epochs(35)
    cnn.set_learning_rate(0.00001)
    cnn.compile_model()
    cnn.train_model()
    cnn.save_model()
    '''
    flt = FaultSimulation()
   # flt.classify_on_bulk("C:\\Users\\Edwin\\Desktop\\Career\\Project\\tester.npz", 'r')
    flt.classify_on_single("C:\\Users\\Edwin\\Desktop\\Career\\Research\\Transformer Inter-turn\\images\\")
    


    


main()



                

