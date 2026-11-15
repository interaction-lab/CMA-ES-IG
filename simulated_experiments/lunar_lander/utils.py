import torch
import torch.nn as nn
import numpy as np
import pandas as pd

#defining our model for selecting actions
class LanderAction(nn.Module):
    def __init__(self):
        super(LanderAction, self).__init__()
        self.regressor = torch.nn.Sequential(
                            torch.nn.Linear(8, 4),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(4, 2),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(2, 2),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(2, 4),
                        )
    def forward(self, x):
        output = self.regressor(x)
        return output
    
# defining our model for predicting measures from parameters
class LanderMeasureSpace(nn.Module):
    def __init__(self, input_size, dim_embedding):
        super(LanderMeasureSpace, self).__init__()
        self.regressor = torch.nn.Sequential(
                            torch.nn.Linear(input_size, dim_embedding),
                            torch.nn.Tanh()
                        )
    def forward(self, x):
        output = self.regressor(x)
        return output
    

def count_parameters(model):
    #returns the number of TRAINABLE parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_partition_list(model):
  '''
  returns the indices at which to slice a list if you want to reshape
  a long array into the parameters for the model

  e.g., your network has the following layers:
        1x2 FF (2 weight terms, 2 bias terms)
        2x2 FF (4 weight terms, 2 bias terms)
        2x1 FF (2 weight terms, 1 bias term)
    then this network has 13 parameters, and if a length 13 list is supplied,
    it should be cut at indices [2,4,8,10,12]
  '''
  partition_list = [0]

  for p in [p for p in model.parameters() if p.requires_grad]:
    partition_list.append(p.numel() + partition_list[-1])

  return partition_list[1:-1]

def vector_to_model(vector):
  '''
  takes in a vector with length==# model parameters and a model and loads the vector into the model
  (this function only works for our specific above model hehe)
  '''
  model = LanderAction()
  new_dict = {}
  sizes = get_partition_list(model)
  for key, value in zip(model.state_dict().keys(), np.split(vector, sizes)):
    new_dict[key] = torch.tensor(value).reshape(*model.state_dict()[key].shape)

  model.load_state_dict(new_dict)
  return model


def load_archive_as_np(dim_embedding, run):
    df = pd.read_csv(f"./data/embeddings/lunar_lander_dim{dim_embedding}_run{run}.csv")
    # print(df)
    return df[[f'measures_{i}' for i in range(dim_embedding)]].to_numpy().T, df[['objective']].to_numpy().flatten()

def calculate_reward_alignment(estimated_preference, true_preference):
    return np.dot(estimated_preference, true_preference) / (np.linalg.norm(estimated_preference) * np.linalg.norm(true_preference))

def calculate_regret(estimated_preference, true_preference, dataset):
    true_utilities = dataset @ true_preference
    estimated_utilities = dataset @ estimated_preference

    true_best = np.max(true_utilities)
    estimated_best = np.max(true_utilities[np.argmax(estimated_utilities)])

    regret = (true_best - estimated_best) / true_best
    return regret

def calculate_query_qualities(true_preference, query):
    utilities = query @ true_preference
    return np.max(utilities), np.mean(utilities), np.median(utilities), np.min(utilities)



def get_features(dim_embedding, emb_dir="./data/embeddings/", type='pca'):
    run = np.random.randint(5)  # assuming there are 5 runs, change as needed
    df = pd.read_csv(f"{emb_dir}/lunar_lander_dim{dim_embedding}_run{run}.csv")
    data = df[[f'measures_{i}' for i in range(dim_embedding)]].to_numpy()

    return data, None # TODO: potentially include the objective measure as one of the features
