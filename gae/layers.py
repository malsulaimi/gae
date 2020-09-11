from __future__ import print_function
from gae.initializations import *
from gae.raggedHelper import * 
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
#tf.disable_v2_behavior()

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}




def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)



class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.tanh, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")+0
        self.dropout = dropout
        self.adj = adj
        self.act = act

    

    def _call(self, inputs):
        x = inputs
        #with tf.control_dependencies([x[0] , x[1] , x[2]]) : 
        #    dummy = multiplyragged_withDesne(x,self.vars['weights'])
        with tf.control_dependencies([x[0] , x[1] , x[2]]) : 
            xdropout = multiplyraggedDropout(x,1.0)
        with tf.control_dependencies([xdropout[0] , xdropout[1] , xdropout[2]]) : 
            x1 = multiplyragged_withDesne(xdropout,self.vars['weights'])#tf.map_fn(self._scan_wight_multiply , x  ) # x[Batch , Node , 32 ] X weights[32,16] = x[Batch , Node , 16 ] =  output[Batch , Node , 16 ]
        with tf.control_dependencies([x1[0] , x1[1] , x1[2]]) :
            x2= multiply2n_ragged(self.adj, x1)    #tf.map_fn(matrix_mat_mul_3d,(self.adj, x)) #ADJ[Batch , Node , Node ] X x[Batch , Node , 16 ] = output[Batch ,Node, 16]
        print("shape of Embedding : " , x )  
        with tf.control_dependencies([x2[0] , x2[1] , x2[2]]) :
            outputs = outputs = self.act(x2[0]) , x2[1] , x2[2] #self.act(x)
    
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.tanh, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero


    
    def _call(self, inputs):
        x = inputs
        print("step 1 : " , x)
       
        ''' Commented by Mousa 
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x) ''' 
        ''' Added by mousa ''' 
        #x = tf.map_fn( self._scan_dropout , x  ) 
        print("step 2 ( after dropout : " , x)
        
        #x = tf.map_fn(self._scan_wight_multiply , x   ) #   x[Batch , Node , Feature ] X Wight[ Feature , 32 ] = output[Batch , Node , 32 ]
        #dummy = multiplyragged_withDesne(x,self.vars['weights']) #### This is a hack an a dummy operation , without it the optimizer fails 
        with tf.control_dependencies([x[0] , x[1] , x[2]]) : 
            xdropout = multiplyraggedDropout(x,1.0)
        with tf.control_dependencies([xdropout[0] , xdropout[1] , xdropout[2]]) : 
            x1= multiplyragged_withDesne(xdropout,self.vars['weights'])
            #print("This is after mat scan : " , x)
        #    x1data = x1[0]#tf.Print(x1[0],[x1[0]] , "This is the input of the convloution")
        #    xp=x1data,x1[1],x1[2]
        
        with tf.control_dependencies([x1[0] , x1[1] , x1[2]]) : 
            x2=multiply2n_ragged(self.adj,x1)#  tf.map_fn(matrix_mat_mul_3d,(self.adj, x1)) # , Adj[ Batch, Node , Node ]  X x[Batch , Node , 32 ]  , output shape = [Batch , Node , 32 ]
            #print("This is after mat multi : " , x)
        with tf.control_dependencies([x2[0] , x2[1] , x2[2]]) : 
            outputs = self.act(x2[0]) , x2[1] , x2[2] # tf.map_fn(self._scan_activation , x   )##self.act(x)
        #shape Batch X  16
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
   
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        #inputs = tf.nn.dropout(inputs, 1-self.dropout)          #input is [Batch ,Node, 16]
        
        #print("This is after mat multi : " , inputs)
        with tf.control_dependencies([inputs[0] , inputs[1] , inputs[2]]) : 
            x = multiplyraggedTranspose(inputs)#tf.map_fn(self._scan_transpose , inputs   )  # X became [Batch ,16, Node]
        
        with tf.control_dependencies([x[0] , x[1] , x[2] , inputs[0],inputs[1],inputs[2]]) : 
            x1=  multiply2n_ragged(inputs,x)#tf.map_fn(matrix_mat_mul_3d,(inputs, x)) #tf.matmul(inputs, x)
        #print("Inner product MAt MUL : " , x ) #   #input is [Batch ,Node, 16] X [Batch ,16, Node] = output[Batch ,Node, Node]
         
        #x = tf.reshape(x, [1000,-1])
        with tf.control_dependencies([x1[0] , x1[1] , x1[2]])  :          
            outputs = self.act(x1[0]) , x1[1] , x1[2]
        print("The final output is : " , outputs)
        
        return outputs
