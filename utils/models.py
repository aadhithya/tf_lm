import tensorflow as tf
import numpy as np
import utils as utils

class Model:

    def __init__(self, name):
        self.name=name

        self.inputs= None
        self.labels=None

        self.lr = None
        self.loss = None
        self.accuracy = None
        self.train_step = None

        self.time_steps = None

        self.train_history = {}


def train_model(model, train_X, train_y,epochs=5, batch_Size=128, print_Every=200, verbose=True):

    if not isinstance(model, Model):
        raise TypeError('model should be of type Model!')

    
    train_losses=[]
    train_accs=[]
    
    with tf.Session() as s:
        
        s.run(tf.global_variables_initializer())


        for e in range(epochs):
            print(f'Epoch {e+1}:')

            for ix,batch_idx in enumerate(utils.get_batch_idx(train_X,batch_Size)):

                _ = s.run([model.train_step],feed_dict={model.inputs:train_X[batch_idx],model.labels:train_y[batch_idx]})
                tr_loss, tr_acc = s.run((model.loss,model.accuracy),feed_dict={model.inputs:train_X[batch_idx],model.labels:train_y[batch_idx]})
                
                train_losses+=[tr_loss]
                train_accs+=[tr_acc]

                if ix % print_Every == 0 and verbose:
                    print(f'Step {ix}: Training Loss: {tr_loss:.3f}, Training Accuracy: {tr_acc:.3f}')
        
        model.train_history['training_losses'] = train_losses
        model.train_history['training_accuracies'] = train_accs
    









class SingleLSTMLayerModel(Model):

    def __init__(self, name):
        super().__init__(name)

    def build (self):
        pass