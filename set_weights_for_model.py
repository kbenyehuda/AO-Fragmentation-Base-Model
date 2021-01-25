import pickle

def set_weights(weights_pickle,model):
    pickle_in = open(weights_pickle, "rb")
    example_dict = pickle.load(pickle_in)
    weights = example_dict[list(example_dict.keys())[0]]

    model.set_weights(weights)
    return model
