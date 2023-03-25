import keras.backend as K



def summarize_tf_model(tf_model):
    print('learning rate:', K.eval(tf_model.optimizer.lr))

    for idx, l in enumerate(tf_model.layers, start=1):

        try:
            print('Layer {idx}'.format(idx=idx), l.activation)
            print('Number of Neurons:', l.units)

        except: # some layers don't have any activation
            pass