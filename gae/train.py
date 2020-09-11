from __future__ import division
from __future__ import print_function
from sklearn.preprocessing import normalize
import time
import os
import sys
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
from tqdm import tqdm 
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
#tf.disable_v2_behavior()
from tensorflow.python import debug as tf_debug
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
codebase = 'C:\\Users\\USER\\Documents\\Projects\\MastersEnv\\GraphAutoEncoder\\'
sys.path.append(codebase)
from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges , construct_feed_dict_dummy

def calculate_auc_ap(labels,predictions):
    
    casted_preds = []
    casted_labels = []
    for y_h , y in zip(predictions,labels) : 
        #casted_preds = casted_preds + [bool(y_h)]
        casted_labels=casted_labels + [bool(y)]
        
    #print(casted_labels)
    #print(labels)
    roc_score = roc_auc_score(casted_labels, predictions )
    ap_score = average_precision_score(casted_labels, predictions)
    
    return roc_score , ap_score



def getData2(array1,array2,iteration , batch_size):
    '''
    check the size of the dataset , floor(len(adj_arr)/batch_size) = NumberOFBatchs 
    iteration 0  , start = 0 , end = batch_Size
    iteration 1 , start= batch_size , end = 2Xbatch_size
    iteration 2 , start= 2Xbatch_size , end = 3Xbatch_size
    .
    .
    .
    .
    iteration dataset , start = (iterations*batch_size)%len(adj_arr) , end = iterations*batch_size+batch_Size
    
    '''
    if ((iteration*batch_size)%len(array1)+batch_size) < len(array1) :
        start = (iteration*batch_size)%len(array1)
        end = (iteration*batch_size)%len(array1)+batch_size
    else :
        start =0  
        end = batch_size #iteration % size 
        print('************************************************************************************************************************RESET********************************************************************************************')
       
    print('processing from ' , start  , 'to : ' , end) 
    return fetch_data3(array1[start:end],array2[start:end],0,0)#,[]  , [] ,adj_norm_arr[start:end] , adj_label_arr[start:end] , features_arr[start:end]







batchsize= 50


def generateL1Listwriteback_numpy(start_offest,step,num):
    counter=0
    values = [] 
    while counter*step <=num*step :  
        values = values+[(counter*step)+start_offest]
        counter = counter+1
    return values



def fetch_data3(array1,array2,fixedSizeRow,fixedSizeWidth): 
    feat_array = []
    label_array = []
    File_locations="C:\\Users\\USER\\Documents\\Projects\\MastersEnv\\GraphAutoEncoder\\gae_batch_update\\myData"
    files = os.listdir("C:\\Users\\USER\\Documents\\Projects\\MastersEnv\\GraphAutoEncoder\\gae_batch_update\\myData")
    #print('creating arrays')
    for file in array1: 
        temp1 = np.load(File_locations+'\\'+file ,allow_pickle=True , encoding='bytes' )[()]  
        temp1_padded = temp1
        if fixedSizeRow > 0 : 
            rowspad = fixedSizeRow-temp1.shape[0]
            colspad = fixedSizeWidth-temp1.shape[1]     
            temp1_padded = np.pad(temp1,((0,rowspad),(0,colspad)) , 'constant')
        #print("ADJ before padding " , temp1.shape , " After Padding " , temp1_padded.shape)
        label_array =  label_array + [temp1_padded ] 
    for file in array2: 
        temp2=np.load(File_locations+'\\'+file ,allow_pickle=True , encoding='bytes' )[()]  
        temp2_padded = temp2
        if fixedSizeRow > 0 :
            rowspad = fixedSizeRow-temp2.shape[0]
            temp2_padded =  np.pad(temp2,((0,rowspad),(0,0)) , 'constant')
        #print("featuer before padding " , temp2.shape , " After Padding " , temp2_padded.shape)
        feat_array =  feat_array + [ temp2_padded] 
        #print('here')
    
    adj_arr = [] 
    adj_orig_arr = []
    adj_norm_arr = []
    adj_label_arr=[]
    features_arr=[]
    adj_total_split = [0] 
    adj_orig_total_split = [0]
    adj_norm_total_split = [0]
    adj_label_total_split=[0]
    features_total_split=[0]
    adj_total_split_0 = [0] 
    adj_orig_total_split_0 = []
    adj_norm_total_split_0 = []
    adj_label_total_split_0=[]
    features_total_split_0=[]    
    adj_curr_split = 0 
    adj_orig_curr_split = 0
    adj_norm_curr_split = 0
    adj_label_curr_split=0
    features_curr_split=0
    pos_weight =0
    norm = 0.0
    index = 0.0 
    for test1 , test2 in (zip(label_array,feat_array)) :
        ######### start core pre-processing
        
        adj = csr_matrix(test1)
        features=csr_matrix(test2)
        #features = sparse_to_tuple(features.tocoo())
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()
        #adj_orig = sparse_to_tuple(adj_orig.tocoo())
        adj_norm   = preprocess_graph(adj)
        adj_label = adj + sp.eye(adj.shape[0])
        #adj_label = sparse_to_tuple(adj_label)
        if (float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) != 0) and (adj.sum()!=0) : ## avoid diving by Zero and skip currupted data 
            pos_weight = pos_weight+ ( float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() )
            norm = norm+(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))
        else : 
            continue
        ########## finished core pre-processing 
        tmp1 = adj_label.todense().shape
        #print("adj_label shape : " , tmp1)
        adj_label_total_split_0=adj_label_total_split_0 + generateL1Listwriteback_numpy(len(adj_label_arr),tmp1[1],tmp1[0])
        #print("L2 split : " , adj_label_total_split_0)
        #print("data values :" ,len(adj_label_arr ) )
        adj_label_arr  = adj_label_arr + np.reshape(adj_label.todense(),-1).tolist()[0]
        adj_label_curr_split = adj_label_curr_split + len(adj_label.todense().tolist())
        adj_label_total_split = adj_label_total_split+[adj_label_curr_split]
        #print("L1 split : " , adj_label_total_split)
        
        tmp1 = features.todense().shape
        #print("features shape : " , tmp1)
        features_total_split_0=features_total_split_0 + generateL1Listwriteback_numpy(len(features_arr),tmp1[1],tmp1[0])
        #print("L2 split : " , features_total_split_0)
        #print("data values :" ,len(features_arr ))
        features_arr = features_arr+ np.reshape(features.todense(),-1).tolist()[0]
        #print("data values :" ,features_arr )
        features_curr_split = features_curr_split + len(features.todense().tolist())
        features_total_split = features_total_split+[features_curr_split]
        #print("L1 split : " , features_total_split)
        
        tmp1 = adj_norm.todense().shape
        #print("adj_norm shape : " , tmp1)
        adj_norm_total_split_0=adj_norm_total_split_0 + generateL1Listwriteback_numpy(len(adj_norm_arr),tmp1[1],tmp1[0])
        #print("L2 split : " , adj_norm_total_split_0)
        #print("data values :" ,len(adj_norm_arr ))
        adj_norm_arr = adj_norm_arr + np.reshape(adj_norm.todense(),-1).tolist()[0]
        #print("data values :" ,adj_norm_arr )
        adj_norm_curr_split = adj_norm_curr_split + len(adj_norm.todense().tolist())
        adj_norm_total_split = adj_norm_total_split+[adj_norm_curr_split]
        #print("L1 split : " , adj_norm_total_split)
        
        tmp1 = adj_orig.todense().shape
        #print("adj_orig shape : " , tmp1)
        adj_orig_total_split_0=adj_orig_total_split_0 + generateL1Listwriteback_numpy(len(adj_orig_arr),tmp1[1],tmp1[0])
        #print("L2 split : " , adj_orig_total_split_0)
        #print("data values :" ,len(adj_orig_arr ))
        adj_orig_arr = adj_orig_arr + np.reshape(adj_orig.todense(),-1).tolist()[0]
        #print("data values :" ,adj_orig_arr )
        adj_orig_curr_split = adj_orig_curr_split + len(adj_orig.todense().tolist())
        adj_orig_total_split = adj_orig_total_split+[adj_orig_curr_split]
        #print("L1 split : " , adj_orig_total_split)
    adj_orig_total_split_0 = list(set(adj_orig_total_split_0))
    adj_norm_total_split_0 = list(set(adj_norm_total_split_0))
    adj_label_total_split_0 = list(set(adj_label_total_split_0))
    features_total_split_0 = list(set(features_total_split_0))
    adj_orig_total_split_0.sort()
    adj_norm_total_split_0.sort()
    adj_label_total_split_0.sort()
    features_total_split_0.sort()
    adj_orig_arr_f = adj_orig_arr,adj_orig_total_split_0,adj_orig_total_split
    adj_norm_arr_f = adj_norm_arr , adj_norm_total_split_0 , adj_norm_total_split
    adj_label_arr_f = adj_label_arr , adj_label_total_split_0 , adj_label_total_split
    features_arr_f = features_arr,features_total_split_0 , features_total_split
    #adj_orig_arr_f = adj_orig_arr,adj_orig_total_split_0,adj_orig_total_split
    #adj_norm_arr_f = adj_norm_arr , adj_norm_total_split_0 , adj_norm_total_split
    #adj_label_arr_f = adj_label_arr , adj_label_total_split_0 , adj_label_total_split
    #features_arr_f = features_arr,features_total_split_0 , features_total_split
    pos_weight =pos_weight/len(label_array)
    norm = pos_weight/len(label_array)
    '''
    return adj_arr  , tf.ragged.RaggedTensorValue(np.array(adj_orig_arr),np.array(adj_orig_total_split))  \
            ,tf.ragged.RaggedTensorValue(np.array(adj_norm_arr),np.array(adj_norm_total_split)) \
            ,tf.ragged.RaggedTensorValue(np.array(adj_label_arr),np.array(adj_label_total_split)) \
            ,tf.ragged.RaggedTensorValue(np.array(features_arr),np.array(features_total_split))
    '''
    return pos_weight , norm , adj_arr , adj_orig_arr_f , adj_norm_arr_f , adj_label_arr_f , features_arr_f




# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 20, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 10, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset


if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless


#added by mousa 
placeholders = {
    #'features': tf.ragged.placeholder(dtype=tf.float32 ,ragged_rank=1)   ,
    #'adj': tf.ragged.placeholder(dtype=tf.float32,ragged_rank=1 ) ,
    #'adj_orig': tf.ragged.placeholder(dtype=tf.float32,ragged_rank=1 )  ,
    #'dropout': tf.placeholder_with_default(0., shape=()) }
    'features_values' : tf.placeholder(dtype=tf.float32),
    'features_l2_splits' : tf.placeholder(dtype=tf.int32),
    'features_l1_splits': tf.placeholder(dtype=tf.int32) ,
    'adj_values' : tf.placeholder(dtype=tf.float32),
    'adj_l2_splits' : tf.placeholder(dtype=tf.int32),
    'adj_l1_splits': tf.placeholder(dtype=tf.int32) ,
    'adj_orig_values' : tf.placeholder(dtype=tf.float32),
    'adj_orig_l2_splits' : tf.placeholder(dtype=tf.int32),
    'adj_orig_l1_splits': tf.placeholder(dtype=tf.int32) ,
    'dropout': tf.placeholder_with_default(0., shape=()) , 
    'pos_weight' : tf.placeholder(dtype=tf.float32) , 
    'norm' : tf.placeholder(dtype=tf.float32)  ,
    'ROC_Score' : tf.placeholder(dtype=tf.float32)  ,  
    'AP' : tf.placeholder(dtype=tf.float32)    }
#(?, 3703, 32)


# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, 300, 0)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)





# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        labels_placeholders = placeholders['adj_orig_values'] , placeholders['adj_orig_l2_splits'] , placeholders['adj_orig_l1_splits']
        opt = OptimizerAE(preds=model.reconstructions,batch_size=batchsize,
                          labels=labels_placeholders
                          ,pos_weight=placeholders['pos_weight'],
                          norm=placeholders['norm'] , roc=placeholders['ROC_Score'] , ap=placeholders['AP'])                          
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig_values'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))   #sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(tf.global_variables_initializer())
 
files = os.listdir("C:\\Users\\USER\\Documents\\Projects\\MastersEnv\\GraphAutoEncoder\\gae_batch_update\\myData")
array1 = []
array2 = []
limit = 100000
limit_counter  = 0
for file in tqdm(files) : 
    if file.endswith(".adj.npy"): 
        array1 = array1 + [file]
        limit_counter = limit_counter +1
    if file.endswith(".feat.npy"):    
        array2 = array2 + [file]
    if  limit_counter== limit   :
        break 

summ_writer = tf.summary.FileWriter('C:\\Users\\USER\\Documents\\Projects\\MastersEnv\\GraphAutoEncoder\\gae\\summaries', sess.graph)
#adj_arr , adj_orig_arr , adj_norm_arr , adj_label_arr,features_arr =load_Data3(fixedSizeRow,fixedSizeWidth)
#C:\Users\USER\Documents\Projects\MastersEnv\GraphAutoEncoder\gae\checkpoints
checkpointpath= 'C:\\Users\\USER\\Documents\\Projects\\MastersEnv\\GraphAutoEncoder\\gae\\checkpoints\\model_all_10.ckpt'
saver = tf.train.Saver()

if os.path.exists(checkpointpath+'.index') : 
    saver.restore(sess, checkpointpath) ## restoring check point 
    print("Restoring data from :" , checkpointpath)

count = 0 
offset = 0  ## this is the last training sample recorded 
#count = offset % 10 +1
for epoch in tqdm(range(100)):

    t = time.time()
    for x in tqdm(range(100000)) : 
        #print("IN IN IN ")
        ind =  x + offset 
        pos , norm ,_ , _ , adj_norm_batch , adj_label_batch,features_batch =  getData2(array1 , array2, ind , batchsize)
        #feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict =construct_feed_dict(adj_norm_batch, adj_label_batch, features_batch , pos , norm, placeholders)
        # Run batch weight update
        #print("**************************************************************************")
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy ], feed_dict=feed_dict)
        avg_cost = outs[1]
        avg_accuracy = outs[2]
        emb ,output2= sess.run([model.z_mean , model.reconstructions] , feed_dict=feed_dict)
        
        if (ind % 10 ==0) & (ind !=0)  : 
            save_path = saver.save(sess, checkpointpath) ## Saving check point 
            print("Model saved in path: %s" % save_path)
        
        if ind % 10 == 0 : 
            print('index' , ind ) 
            print('count' , count)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),"train_acc=", "{:.5f}".format(avg_accuracy), "time=", "{:.5f}".format(time.time() - t))
            correct_prediction = sess.run(opt.final_output, feed_dict=feed_dict)
            print("Array 1 : " , len(list(correct_prediction)))
            print("Array 2 : " , len(adj_label_batch[0]))
            roc_score,ap_score = calculate_auc_ap(adj_label_batch[0], list(correct_prediction) )
            print('Test ROC score: ' + str(roc_score))
            print('Test AP score: ' + str(ap_score))
            feed_dict.update({placeholders['ROC_Score']: roc_score})
            feed_dict.update({placeholders['AP']: ap_score})
            summ = sess.run(opt.summary, feed_dict=feed_dict)
            feed_dict.update({placeholders['ROC_Score']: 0})
            feed_dict.update({placeholders['AP']: 0})            
            #summ_writer.add_summary(summ,(x+1)+(epoch*len(adj_norm_batch)))            
            summ_writer.add_summary(summ,count)
            #print("length of the embedding output is " , len(emb))
            print("length of the embedding output is output  :  " , len(emb[0]))
            #print(emb[0])
            count = count +1
            print("DO some thing")
            
             
            '''
            for a , b in zip(output2[0] , adj_label_batch[0] ) : 
                print("real ===="  , a)
                print("reconstruction  ===="  , b)
            '''
            
        
    feed_dict.update({placeholders['dropout']: 0})
for x in emb : 
    print(np.reshape(x,[-1]))
    print("============================================== New array =======================================")
    # Compute average loss


    #roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    #val_roc_score.append(roc_curr)



print("Optimization Finished!")

