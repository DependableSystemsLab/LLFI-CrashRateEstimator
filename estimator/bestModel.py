class BestModelObject:
#just a container for a few lists and other objects

    def __init__(self, model, scores, hyperparameters):
        self.model = model
        self.scores = scores
        self.hyperparameters = hyperparameters

