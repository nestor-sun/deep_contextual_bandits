import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import warnings
df = pd.read_csv('C:/Users/Varun Suryan/Desktop/ML Guarnatess/Project/data/mushrooms.csv')
warnings.filterwarnings("ignore")

K = 50

# lablels of mushrooms
y = df['class'].astype("category").cat.codes

# charactersitcis
X = pd.get_dummies(df.drop('class', 1), drop_first = True)



# determine the supported device
def get_device():
    
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    
    return torch.device('cpu') # don't have GPU 
    

# initializing the neural netowork
eaten = 10
eaten_X = X.sample(eaten)

eaten_y = y[eaten_X.index]
X, y = X.drop(eaten_X.index), y.drop(eaten_X.index)

class Net(nn.Module):
    def __init__(self, input_dim):
      self.input_dim = input_dim
      super(Net, self).__init__()

      # First fully connected layer
      self.fc1 = nn.Linear(self.input_dim, 128)
      
      self.dropout1 = nn.Dropout2d(0.25)

      # Second fully connected layer that outputs our label
      self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
    	x = self.fc1(x)
    	x = F.relu(x)
    	x = self.dropout1(x)
    	x = self.fc2(x)
    	return x


random_data = torch.rand((1, X.shape[1]))
model = Net(X.shape[1])

learning_rate = 1e-4
loss_fn = nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def df_to_tensor(df):
   	return torch.from_numpy(df.values).float()


for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(df_to_tensor(eaten_X))
    loss = loss_fn(y_pred, df_to_tensor(eaten_y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# values, indices = model(df_to_tensor(X)).topk(K, 0)

for _ in range(100):

  for t in range(1000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(df_to_tensor(eaten_X))
    loss = loss_fn(y_pred, df_to_tensor(eaten_y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  values = model(df_to_tensor(X))
  eaten = pd.Series(values[:, 0].detach().numpy(), index = X.index).sort_values(ascending = False).head(K).index
  eaten_X = pd.concat([eaten_X, X.loc[eaten]])
  eaten_y = pd.concat([eaten_y, y.loc[eaten]])
  print(eaten_y.sum())
  X, y = X.drop(X.loc[eaten, :].index), y.drop(X.loc[eaten, :].index)


# print(df_to_tensor(y)[indices])
# eaten_X = X.sample(K)

# eaten_y = y[eaten_X.index]
# X, y = X.drop(eaten_X.index), y.drop(eaten_y.index)


class mushrooms:
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def eaten(self, indices):
		self.indices = indices