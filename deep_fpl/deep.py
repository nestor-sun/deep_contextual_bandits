"""Deep environments and bandit algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import platform
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

#from google3.pyglib import gfile


class WheelBandit(object):
  """Binary classification bandit."""

  def __init__(self, mu1=1.2, mu2=1.0, mu3=50.0, sigma=0.01, delta=0.5):
    self.mu = np.asarray([mu1, mu2, mu3]) / mu3
    self.sigma = sigma
    self.delta = delta
    self.K = 5

    self.randomize()

  def randomize(self):
    # random 2-dimensional vector of length [0, 1]
    v = np.random.randn(2)
    v *= np.random.rand() / np.linalg.norm(v)

    self.X = np.zeros((self.K, 3 * self.K))
    for i in range(self.K):
      self.X[i, 3 * i : 3 * i + 2] = v
      self.X[i, 3 * i + 2] = 1

    self.rt = np.zeros(self.K)
    self.rt[0] = self.mu[0] + self.sigma * np.random.rand()
    self.rt[1 :] = self.mu[1] + self.sigma * np.random.rand(self.K - 1)
    if np.linalg.norm(v) > self.delta:
      self.rt[2 * (v[0] > 0) + (v[1] > 0) + 1] = \
        self.mu[2] + self.sigma * np.random.rand()

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous REWARD of the arm
    return self.rt[arm]

  def print(self):
    return "Wheel bandit"


class BinaryClassBandit(object):
  """Binary classification bandit."""

  def __init__(self, X, y, pos_label=1, K=2, pos_prob=1.0):
    self.Xall = X
    self.K = K
    self.pos_prob = pos_prob

#     if X.shape[1] >= 10:
#       d = int(np.sqrt(X.shape[1]))
#       cols = np.random.permutation(X.shape[1])[: d]
#       self.Xall = X[:, cols]
#
#       print("%d randomly chosen features" % d)

    self.rall = np.zeros(y.shape)
    self.rall[y == pos_label] = 1

    self.pulled_arms = - np.ones(100000)
    self.log_ndx = 0
    self.log_next_pull = False

    self.randomize()

  def randomize(self):
    self.imgs = np.random.randint(self.Xall.shape[0], size=self.K)
    self.X = self.Xall[self.imgs, :]
    self.rt = self.rall[self.imgs]

    self.is_pos_arm = self.rt.sum() > 0
    if self.is_pos_arm:
      self.pos_arm = np.flatnonzero(self.rt)[0]
    else:
      self.pos_arm = np.random.randint(self.K)

    self.rt = (np.random.rand(self.K) < self.pos_prob) * self.rt + \
      (np.random.rand(self.K) < 1 - self.pos_prob) * (1 - self.rt)

    self.log_next_pull = True

  def reward(self, arm):
    if self.log_next_pull:
      if self.is_pos_arm:
        self.pulled_arms[self.log_ndx] = self.imgs[arm]
      self.log_ndx += 1
      self.log_next_pull = False

    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous REWARD of the arm
    return self.rt[arm]

  def print(self):
    return "Binary classification bandit"


def pixelize(X, pixel=2):
  Xp = np.zeros((X.shape[0], X.shape[1] // pixel, X.shape[2] // pixel))
  for i in range(X.shape[1]):
    for j in range(X.shape[2]):
      Xp[:, i // pixel, j // pixel] += X[:, i, j]
  Xp /= pixel * pixel
  return Xp


def load_dataset(dataset):
  print("python %s" % platform.python_version())
  print("tf %s" % tf.__version__)
  print("keras %s" % keras.__version__)

  print("Preprocessing dataset %s..." % dataset)

  data_dir = "/cns/pw-d/home/bkveton/PHE/Datasets"
  if dataset == "iris":
    with gfile.Open("%s/%s.txt" % (data_dir, dataset)) as f:
      D = np.loadtxt(f, delimiter=",")
    X = D[:, : -1]
    y = D[:, -1]
  elif dataset == "letter_recognition":
    with gfile.Open("%s/%s.txt" % (data_dir, dataset)) as f:
      D = np.genfromtxt(f, dtype="str", delimiter=",")
    X = D[:, 1 :].astype(float)
    y = np.asarray(map(ord, D[:, 0]))
    y = y - y.min()
  elif dataset == "digit_recognition":
    with gfile.Open("%s/%s.txt" % (data_dir, dataset)) as f:
      D = np.loadtxt(f, delimiter=",")
    X = 2 * D[:, : -1] / 16 - 1  # [-1, 1] features
    y = D[:, -1]
  elif dataset == "mnist":
    (X, y), _ = keras.datasets.mnist.load_data()
    X = 2 * X.astype(float) / 255 - 1  # [-1, 1] features
    X = np.reshape(X, (X.shape[0], -1))
  elif dataset == "fashion_mnist":
    (X, y), _ = keras.datasets.fashion_mnist.load_data()
    X = 2 * X.astype(float) / 255 - 1  # [-1, 1] features
    X = np.reshape(X, (X.shape[0], -1))
  elif dataset == "cifar-10":
    with gfile.Open("%s/%s.txt" % (data_dir, dataset)) as f:
      D = np.loadtxt(f, delimiter=" ")
    X = D[:, : -1] / 255
    y = D[:, -1]

  print("%d examples, %d features, %d labels" %
    (X.shape[0], X.shape[1], y.max() + 1))

  return X, y


def perturbed_crossentropy(y_true, y_pred):
  loglik = (1 - y_true) * tf.math.log(1 - y_pred) + \
    y_true * tf.math.log(y_pred)
  return - loglik


class DeepFPL(object):
  """Deep follow the perturbed leader."""

  def __init__(self, env, n, params):
    self.env = env
    self.n = n
    self.K = self.env.X.shape[0]
    self.d = self.env.X.shape[1]

    self.hidden_nodes = 0  # neural network architecture
    self.hidden_activation = "relu"  # activation in the hidden layer

    self.a = 0  # perturbation scale (Gaussian noise)
    self.optimizer = "adam"  # gradient optimizer
    self.lr = 1.0 / np.sqrt(self.n)  # learning rate
    self.init_explore = np.sqrt(self.d)  # initial exploration rounds
    self.batch_size = 32  # mini-batch size
    self.cheat = False

    for attr, val in params.items():
      setattr(self, attr, val)

    # parse neural network architecture
    if isinstance(self.hidden_nodes, int):
      if not self.hidden_nodes:
        self.hidden_nodes = ""
      else:
        self.hidden_nodes = str(self.hidden_nodes)
    self.hidden_nodes = np.fromstring(self.hidden_nodes, dtype=int, sep="-")

    # sufficient statistics
    self.model_size = self.n  # model size
    self.X = np.zeros((self.model_size, self.d))
    self.y = np.zeros(self.model_size)

    # neural net
    self.num_layers = self.hidden_nodes.size
    input = Input((self.d,))

    if not self.num_layers:
      output = Dense(1, activation="sigmoid")(input)
    else:
      hidden = Dense(self.hidden_nodes[0],
        activation=self.hidden_activation)(input)
      for layer in range(1, self.num_layers):
        hidden = Dense(self.hidden_nodes[layer],
          activation=self.hidden_activation)(hidden)
      output = Dense(1, activation="sigmoid")(hidden)

    self.model = Model(inputs=input, outputs=output)
    if self.optimizer == "sgd":
      self.model.compile(loss=perturbed_crossentropy,
        optimizer=keras.optimizers.SGD(learning_rate=self.lr))
    elif self.optimizer == "adam":
      self.model.compile(loss=perturbed_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=self.lr, amsgrad=True))
    elif self.optimizer == "rmsprop":
      self.model.compile(loss=perturbed_crossentropy,
        optimizer=keras.optimizers.RMSprop(learning_rate=self.lr))
    else:
      raise Exception("Unknown optimizer: %s" % self.optimizer)

  def update(self, t, arm, r):
    if self.cheat:
      if np.random.rand() < 0.5:
        arm = self.env.pos_arm
      else:
        arm = np.random.randint(self.K)
      r = self.env.reward(arm)

    self.X[t, :] = self.env.X[arm, :]
    self.y[t] = r

    if t == self.n - 1:
      keras.backend.clear_session()

  def get_arm(self, t):
    # # exponential learning rate decay from self.lr to 1e-4
    # # stabilizes Adam when self.lr is high
    # lrt = self.lr * np.exp(np.log(1e-4 / self.lr) * t / self.n)
    # keras.backend.set_value(self.model.optimizer.learning_rate, lrt)

    if t >= self.batch_size:
      # sub = np.arange(t - self.batch_size, t)
      sub = np.random.randint(t, size=self.batch_size)
      noise = self.a * \
        np.minimum(np.maximum(np.random.randn(sub.size), -6), 6)
      self.model.train_on_batch(self.X[sub, :], self.y[sub] + noise)

    if t >= max(self.batch_size, self.init_explore):
      self.mu = self.model.predict(self.env.X).flatten()
    else:
      self.mu = np.random.rand(self.K)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "DeepFPL"


class NeuralLinear(object):
  """GLM bandit with a learned embedding."""

  def __init__(self, env, n, params):
    self.env = env
    self.n = n
    self.K = self.env.X.shape[0]
    self.d = self.env.X.shape[1]

    self.hidden_nodes = 0  # neural network architecture
    self.hidden_activation = "relu"  # activation in the hidden layer

    self.a = 0  # perturbation scale (Gaussian noise)
    self.optimizer = "adam"  # gradient optimizer
    self.lr = 1.0 / np.sqrt(self.n)  # learning rate
    self.init_explore = np.sqrt(self.d)  # initial exploration rounds
    self.batch_size = 32  # mini-batch size
    self.relepe = self.n // 10  # representation learning period

    for attr, val in params.items():
      setattr(self, attr, val)

    # parse neural network architecture
    if isinstance(self.hidden_nodes, int):
      if not self.hidden_nodes:
        self.hidden_nodes = ""
      else:
        self.hidden_nodes = str(self.hidden_nodes)
    self.hidden_nodes = np.fromstring(self.hidden_nodes, dtype=int, sep="-")

    # sufficient statistics
    self.model_size = self.n  # model size
    self.X = np.zeros((self.model_size, self.d))
    self.y = np.zeros(self.model_size)

    # logistic model
    input = Input((self.hidden_nodes[-1],))
    output = Dense(1, activation="sigmoid")(input)
    self.model = Model(inputs=input, outputs=output)
    self.model.compile(loss=perturbed_crossentropy,
      optimizer=keras.optimizers.SGD(learning_rate=self.lr))

    # embedding
    self.num_layers = self.hidden_nodes.size
    input = Input((self.d,))
    if self.num_layers == 1:
      hidden = Dense(self.hidden_nodes[0],
        activation=self.hidden_activation, name="embedding")(input)
    else:
      hidden = Dense(self.hidden_nodes[0],
        activation=self.hidden_activation)(input)
      for layer in range(1, self.num_layers - 1):
        hidden = Dense(self.hidden_nodes[layer],
          activation=self.hidden_activation)(hidden)
      hidden = Dense(self.hidden_nodes[-1],
        activation=self.hidden_activation, name="embedding")(hidden)
    output = Dense(1, activation="sigmoid")(hidden)

    self.emodel = Model(inputs=input, outputs=output)
    if self.optimizer == "sgd":
      self.emodel.compile(loss="binary_crossentropy",
        optimizer=keras.optimizers.SGD(learning_rate=self.lr))
    elif self.optimizer == "adam":
      self.emodel.compile(loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=self.lr, amsgrad=True))
    elif self.optimizer == "rmsprop":
      self.emodel.compile(loss="binary_crossentropy",
        optimizer=keras.optimizers.RMSprop(learning_rate=self.lr))
    else:
      raise Exception("Unknown optimizer: %s" % self.optimizer)

    self.embed_me = Model(inputs=self.emodel.input,
      outputs=self.emodel.get_layer("embedding").output)

  def update(self, t, arm, r):
    if t < self.relepe:
      self.X[t, :] = self.env.X[arm, :]
      if t == self.relepe - 1:
        # learn embedding
        num_steps = 1000
        for i in range(num_steps):
          sub = np.random.randint(t, size=self.batch_size)
          self.emodel.train_on_batch(self.X[sub, :], self.y[sub])

        # embed history and reshape feature vectors
        eX = self.embed_me.predict(self.X[: t + 1, :])
        self.X = np.zeros((self.model_size, self.hidden_nodes[-1]))
        self.X[: t + 1, :] = eX
    else:
      self.X[t, :] = self.embed_me.predict(self.env.X[arm, :][np.newaxis, :])
    self.y[t] = r

    if t == self.n - 1:
      keras.backend.clear_session()

  def get_arm(self, t):
    if t >= max(self.batch_size, self.relepe):
      # sub = np.arange(t - self.batch_size, t)
      sub = np.random.randint(t, size=self.batch_size)
      noise = self.a * \
        np.minimum(np.maximum(np.random.randn(sub.size), -6), 6)
      self.model.train_on_batch(self.X[sub, :], self.y[sub] + noise)

    if t >= max(max(self.batch_size, self.init_explore), self.relepe):
      eX = self.embed_me.predict(self.env.X)
      self.mu = self.model.predict(eX).flatten()
    else:
      self.mu = np.random.rand(self.K)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "NeuralLinear"
