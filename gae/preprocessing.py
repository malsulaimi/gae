import numpy as np
import scipy.sparse as sp


def sparse_to_tuple( sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    #print(coords.shape[0])
    #print(coords)

    #print(coords_new)
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def construct_feed_dict(adj_normalized, adj, features,  pos , norm ,  placeholders):
    # construct feed dictionary
    feed_dict = dict()
    '''print(len(features))
    print("leng of indices : " , len(features[0])  , " and this is the shape :  " , features[0].shape ) 
    print("leng of values : " , len(features[1])  , " and this is the shape :  " , features[1].shape ) 
    print("leng of shapes : " , len(features[2])   , " and this is the shape :  " , features[2] )     
    print("======================================================================================")
    print("leng of indices : " , len(adj[0])  , " and this is the shape :  " , adj[0].shape ) 
    print("leng of values : " , len(adj[1])  , " and this is the shape :  " , adj[1].shape ) 
    print("leng of shapes : " , len(adj[2])   , " and this is the shape :  " , adj[2] )   
    print("======================================================================================")
    print("leng of indices : " , len(adj_normalized[0])  , " and this is the shape :  " , adj_normalized[0].shape ) 
    print("leng of values : " , len(adj_normalized[1])  , " and this is the shape :  " , adj_normalized[1].shape ) 
    print("leng of shapes : " , len(adj_normalized[2])   , " and this is the shape :  " , adj_normalized[2] )   
    print(len(adj_normalized))
    l1  = []#np.array([features,features,features])
    l2  =np.array([adj_normalized,adj_normalized,adj_normalized])
    l3=np.array([adj,adj,adj])'''
    '''
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
    'dropout': tf.placeholder_with_default(0., shape=()) }'''
    '''feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})'''
    feed_dict.update({placeholders['adj_values']: adj_normalized[0]})
    feed_dict.update({placeholders['adj_l2_splits']: adj_normalized[1]})
    feed_dict.update({placeholders['adj_l1_splits']: adj_normalized[2]})
    
    feed_dict.update({placeholders['features_values']: features[0]})
    feed_dict.update({placeholders['features_l2_splits']: features[1]})
    feed_dict.update({placeholders['features_l1_splits']: features[2]})
    
    feed_dict.update({placeholders['adj_orig_values']: adj[0]})
    feed_dict.update({placeholders['adj_orig_l2_splits']: adj[1]})
    feed_dict.update({placeholders['adj_orig_l1_splits']: adj[2]})
    
    
    feed_dict.update({placeholders['pos_weight']: pos})
    feed_dict.update({placeholders['norm']: norm})   
 
    return feed_dict
    

def construct_feed_dict_dummy(placeholders):
    # construct feed dictionary
    feed_dict = dict()
    '''print(len(features))
    print("leng of indices : " , len(features[0])  , " and this is the shape :  " , features[0].shape ) 
    print("leng of values : " , len(features[1])  , " and this is the shape :  " , features[1].shape ) 
    print("leng of shapes : " , len(features[2])   , " and this is the shape :  " , features[2] )     
    print("======================================================================================")
    print("leng of indices : " , len(adj[0])  , " and this is the shape :  " , adj[0].shape ) 
    print("leng of values : " , len(adj[1])  , " and this is the shape :  " , adj[1].shape ) 
    print("leng of shapes : " , len(adj[2])   , " and this is the shape :  " , adj[2] )   
    print("======================================================================================")
    print("leng of indices : " , len(adj_normalized[0])  , " and this is the shape :  " , adj_normalized[0].shape ) 
    print("leng of values : " , len(adj_normalized[1])  , " and this is the shape :  " , adj_normalized[1].shape ) 
    print("leng of shapes : " , len(adj_normalized[2])   , " and this is the shape :  " , adj_normalized[2] )   
    print(len(adj_normalized))
    l1  = []#np.array([features,features,features])
    l2  =np.array([adj_normalized,adj_normalized,adj_normalized])
    l3=np.array([adj,adj,adj])'''
    '''
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
    'dropout': tf.placeholder_with_default(0., shape=()) }
    vals =np.array([1.0, 2.2  , 1.1 , 4.0, 5.0 , 1.1 , 6.0, 7.0 , 1.1 , 8.0, 9.0 , 1.1 ,10.0, 11.0 , 1.1 ])
    l2_splits = np.array([0,3,6,9,12,15])
    l1_splits = np.array([0, 2, 4  ]) '''
    #vals = np.array([1.0, 2.2  , 1.1 , 4.0, 5.0 , 1.1 , 6.0, 7.0 , 1.1 , 8.0, 9.0 , 1.1 ,10.0, 11.0 , 1.1 ])
    #l2_splits = np.array([0,2,5,8,11,14])
    #l1_splits = np.array([0, 2, 4  ])
    vals =np.array([1.0, 2.2  , 1.1 , 4.0, 5.0 , 1.1 , 6.0, 7.0 , 1.1 , 8.0])
    l2_splits = np.array([0,2,4,6,8,10])
    l1_splits = np.array([0, 2, 4  ]) 
    feed_dict.update({placeholders['adj_values']: vals})
    feed_dict.update({placeholders['adj_l2_splits']: l2_splits})
    feed_dict.update({placeholders['adj_l1_splits']: l1_splits})
    
    feed_dict.update({placeholders['features_values']: vals})
    feed_dict.update({placeholders['features_l2_splits']: l2_splits})
    feed_dict.update({placeholders['features_l1_splits']: l1_splits})
    
    feed_dict.update({placeholders['adj_orig_values']: vals})
    feed_dict.update({placeholders['adj_orig_l2_splits']: l2_splits})
    feed_dict.update({placeholders['adj_orig_l1_splits']: l1_splits})
    return feed_dict    


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
