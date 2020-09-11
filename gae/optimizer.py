import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
#tf.disable_v2_behavior()
flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
 
    def __init__(self, preds,batch_size, labels, pos_weight, norm , roc, ap ):
        
        with tf.control_dependencies([preds[0],preds[1],preds[2] , labels[0],labels[1],labels[2]]): 
            preds_sub  =preds[0] #tf.reshape(preds[0],[-1])
            print("This is the output OF Priduction" , preds )
            labels_sub =labels[0]#tf.reshape(labels[0],[-1])
            loss=norm*tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight)
            self.cost = tf.reduce_mean(loss) #tf.map_fn(self._scan_reduce_mean , loss , fn_output_signature=struct_scalar) 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
            loss_sum = tf.summary.scalar("Loss" , self.cost)
            self.opt_op = self.optimizer.minimize(self.cost)
            
            self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))                           
            self.final_output= tf.sigmoid(preds_sub)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            acc_sum = tf.summary.scalar("accuracy" , self.accuracy)
            ap_summ = tf.summary.scalar("ap" , ap)
            roc_summ=tf.summary.scalar("roc" , roc)
            self.summary = tf.summary.merge([loss_sum,acc_sum ,ap_summ,roc_summ])

class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
