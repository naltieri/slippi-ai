from typing import List, Optional
import sonnet as snt
import tensorflow as tf
from slippi_ai.data import Batch
import pandas as pd
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
      loss, final_states, distances, predictions, next_action = self.policy.loss(
          tm_gamestate, initial_states)
      action_repeat_loss = tf.reduce_mean(distances['action_repeat'])
      raw_loss = tf.add_n(tf.nest.flatten(distances))
      mean_loss = tf.reduce_mean(raw_loss)
      del distances['action_repeat']
      # maybe do this in the Policy?
      counts = tf.cast(tm_gamestate.counts[1:] + 1, tf.float32)
      weighted_loss = tf.reduce_sum(raw_loss) / tf.reduce_sum(counts)
      mult_loss = tf.math.multiply(raw_loss,  counts)
      inner_product_loss = tf.reduce_sum(mult_loss) / tf.reduce_sum(counts)
      # predicted_num_repeats = tf.math.argmax(predictions['action_repeat'], -1)
      predicted_num_repeats = tf.cast(predictions['action_repeat'],tf.int64)
      action_repeat_accuracy = tf.reduce_mean(tf.cast(predicted_num_repeats==next_action['action_repeat'],tf.float32))
      action_repeat_leq = tf.reduce_mean(tf.cast(predicted_num_repeats<=next_action['action_repeat'],tf.float32))
      action_repeat_mean_diff = tf.reduce_mean(tf.cast(tf.abs(predicted_num_repeats - next_action['action_repeat']),tf.float32))
      action_repeat_diff = tf.math.abs(predicted_num_repeats - next_action['action_repeat'])


    stats = dict(
        loss=mean_loss,
        weighted_loss=weighted_loss,
        inner_product_loss=inner_product_loss,
        action_repeat_loss=action_repeat_loss,
        action_repeat_accuracy=action_repeat_accuracy,
        action_repeat_mean_diff=action_repeat_mean_diff,
        action_repeat_diff=action_repeat_diff,
        predicted_num_repeats=predicted_num_repeats,
        action_repeat_leq=action_repeat_leq,
        actual_num_repeats=next_action['action_repeat'],
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
