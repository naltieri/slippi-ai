from typing import List, Optional
import sonnet as snt
import tensorflow as tf
from slippi_ai.data import Batch

from slippi_ai.policies import Policy

def to_time_major(t):
  permutation = list(range(len(t.shape)))
  permutation[0] = 1
  permutation[1] = 0
  return tf.transpose(t, permutation)

# TODO: should this be a snt.Module?
class Learner:

  DEFAULT_CONFIG = dict(
      learning_rate=1e-4,
      compile=True,
      decay_rate=0.,
  )

  def __init__(self,
      learning_rate: float,
      compile: bool,
      policy: Policy,
      optimizer: Optional[snt.Optimizer] = None,
      decay_rate: Optional[float] = None,
  ):
    self.policy = policy
    self.optimizer = optimizer or snt.optimizers.Adam(learning_rate)
    self.decay_rate = decay_rate
    self.compiled_step = tf.function(self.step) if compile else self.step

  def step(self, batch: Batch, initial_states, train=True):
    bm_gamestate, restarting = batch

    # reset initial_states where necessary
    restarting = tf.expand_dims(restarting, -1)
    initial_states = tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.policy.initial_state(restarting.shape[0]),
        initial_states)

    # switch axes to time-major
    tm_gamestate = tf.nest.map_structure(to_time_major, bm_gamestate)

    with tf.GradientTape() as tape:
      loss, final_states, distances = self.policy.loss(
          tm_gamestate, initial_states)

      raw_loss = tf.add_n(tf.nest.flatten(distances))
      mean_loss = tf.reduce_mean(raw_loss)
      # maybe do this in the Policy?
      counts = tf.cast(tm_gamestate.counts[1:] + 1, tf.float32)
      weighted_loss = tf.reduce_sum(raw_loss) / tf.reduce_sum(counts)
      mult_loss = tf.math.multiply(raw_loss,  tf.cast(counts[1:] + 1, tf.float32) + 1)
      inner_product_loss = tf.reduce_sum(mult_loss) / tf.reduce_sum(counts)


    stats = dict(
        loss=mean_loss,
        weighted_loss=weighted_loss,
        inner_product_loss=inner_product_loss,
        distances=distances,
    )

    if train:
      params: List[tf.Variable] = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_names = [v.name for v in self.policy.trainable_variables]
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(mean_loss, params)
      self.optimizer.apply(grads, params)

      if self.decay_rate:
        for param in params:
          param.assign((1 - self.decay_rate) * param)

    return stats, final_states
