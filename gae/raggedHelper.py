import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
#tf.disable_v2_behavior()

def innerloop_processing(begin_index , end_index , input1) : 
  with tf.control_dependencies([begin_index,end_index ,input1[0] , input1[1] , input1[2] , tf.shape(input1[1][end_index]) ]) : 
    row = tf.slice(input1[0] , [input1[1][begin_index]] , [( input1[1][end_index] )- input1[1][begin_index]])
    #tf.print("Indecies : " , begin_index , "to :" , end_index)
    #tf.print("From : " ,  input1[1][begin_index] , "to : " ,  input1[1][end_index] -1  )
    #tf.print("shapes : " , tf.shape(input1[1][end_index]))
    #tf.print("ROW :" , row)
    #tf.print("ROW lenght : " , tf.shape(row)[0] , "Indecies : " , begin_index , "to :" , end_index )
    numberOfRows = end_index - begin_index
    comp = tf.reshape( row, [numberOfRows, -1])
    #tf.print("new component" , comp)
    return comp


def innerloop_processing2(begin_index , end_index , input1 ) : 
    innerloop_counter = begin_index
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    def innerloop_body(counter , begin_index , end_index , input1 , ta) : 
        #tf.print("Slicing from index : " , counter , "to index : " , counter+1 , "index vector : " , input1[1] , "at counter : " , counter ,  " While shape is  : " , tf.shape(input1[1])[0])
        #tf.print("index value from   " , input1[1][counter] , "to value :" , input1[1][counter+1] , input1[0] ,"at counter : " , counter )
        with tf.name_scope("InnerBOdy"):#with tf.control_dependencies([input1[0],input1[1] ]):
            inner_being_index = input1[1][counter]
            inner_end_index = input1[1][counter+1]
        row = tf.slice(input1[0] , [inner_being_index] ,  [inner_end_index-inner_being_index])
        ta = ta.write(counter-begin_index , row)
        counter = counter + 1 
        return counter , begin_index , end_index , input1 , ta
    
    
    def innerloop_cond(counter , begin_index , end_index , input1 , ta ) : 
        #tf.print("Evaluating condition at endIndex :"  , end_index , " While shape is  : " , tf.shape(input1[1])[0])
        with tf.name_scope("InnerCondition"):##with tf.control_dependencies([input1[0] , input1[1] , input1[2] ,counter, end_index ]):
            return input1[1][counter] < input1[1][end_index] -1  #stop at the next pointer of the l2_splits 

    results = tf.while_loop(innerloop_cond , innerloop_body , [innerloop_counter , begin_index , end_index , input1 , ta] , back_prop=True )
    #print_resutls = tf.print("this is the component result  :" , results[4].stack())
    return results[4].stack()


def generateL1Tensor_writeback(start_offest,step,num):
    counter=tf.constant(0,tf.int32)
    values = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    def cond(values , start_offest , num ,counter) : 
        return counter*step <= num*step
    def body(values , start_offest , num ,counter) : 
        values = values.write(counter,[(counter*step)+start_offest])
        counter = counter+1
        return  values , start_offest , num ,counter
    
    final_values , _ , _ , counter  = tf.while_loop(cond,body,[values , start_offest , num , counter] , back_prop=True)
    with tf.control_dependencies([counter]) :#name_scope('writeback') :
        final = tf.reshape(final_values.stack() , [-1] ) ## avoiding concat following workaround mentioned in https://github.com/tensorflow/tensorflow/issues/34744
    #print_line = tf.print(" xxxxx This is the is the split : " ,  final)
    return final

def multiply2n_ragged(tensor1 , tensor2) : 
    #this  function multiplies two ragged tesnsors of rank 2 . the most outer ranks of the two tensros must be equal .
    #setting variables and constats 
    outerloop_counter = tf.constant(0 , dtype=tf.int32)
    carry_on = tf.constant(0 , dtype=tf.int32)
    taValues = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL2Splits = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL1Splits = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL1Splits = taL1Splits.write(0,[0]) ## required intialization for L1 split only
    innerloop_processing_graphed = tf.function(innerloop_processing)
    generateL1Tensor_writeback_graphed = tf.function(generateL1Tensor_writeback)
    def outerloop_cond(counter,input1,input2 ,taValues  ,taL2Splits , taL1Splits , carry_on ) :
        value = tf.shape(input1[2])[0]-1
        return counter < value ## this is the length of the outermost dimision , stop of this 
    def outloop_body(counter,input1,input2, taValues  ,taL2Splits , taL1Splits , carry_on) : 
        l1_comp_begin = input1[2][counter]                  ## this is begin position of the current row in the outer split  ( ie. the ith value in the outer row split tensor ) 
        l1_comp_end = input1[2][counter+1]                  ## this is end position of the current row in the outer split   (ie. the ith + 1 value in the outer row split tensor)
        l1_comp2_begin = input2[2][counter]                 ## we do the same for the second components 
        l1_comp2_end = input2[2][counter+1]                 ## we do the same for the second components
        comp  = innerloop_processing_graphed(l1_comp_begin ,l1_comp_end ,input1  ) ## now retrive the data to be procesed for the selected rows from vector1
        comp2  =innerloop_processing_graphed(l1_comp2_begin ,l1_comp2_end ,input2  ) ## do the same for vector 2 
        
        #comp2 = tf.transpose(comp2) ### desired operation
        multiply =tf.matmul(comp , comp2) #### This is the desired operation  
        
        
        myshape= tf.shape(multiply) ## calculate the shape of the result in order to prepare to write the result in a ragged tensor format. 
        offset = tf.cond( taValues.size() >0  ,lambda: tf.shape(taValues.concat())[0] , lambda : 0) ### this is a hack, TensorArray.concat returns an error if the array is empty. Thus we check before calling this. 
        #print11=tf.print("=================Final Shape is : " ,myshape[0] , " X " ,myshape[1] )
        l2v = generateL1Tensor_writeback_graphed(offset,myshape[1],myshape[0])  # generate the inner row split of the result for the current element
        taL2Splits=taL2Splits.write(counter,l2v) # write back the inner rowlplit to a TensorArray 
        taValues=taValues.write(counter,tf.reshape(multiply , [-1])) # wirte back the actual ragged tensor elemnts in a another TensorArray
        carry_on=carry_on+myshape[0] ## required to calculate the outer row splite
        taL1Splits=taL1Splits.write(counter+1,[carry_on]) ## This is the outmost row split. 
        with tf.control_dependencies([comp , comp2 ,myshape ,l2v , carry_on , multiply ]):
            counter = counter+1
        return counter , input1,input2, taValues  ,taL2Splits , taL1Splits , carry_on
    with tf.name_scope("RaggedMultiply") :       
        outerloop_finalcounter , _ , _ , ta1,ta2,ta3,_ = tf.while_loop(outerloop_cond,outloop_body,[outerloop_counter , tensor1 , tensor2 ,taValues  ,taL2Splits , taL1Splits,carry_on] ,back_prop=True)
    uinquie_ta2 , _ = tf.unique(ta2.concat())  # this is required since some values might be duplicate in the row split itself 
    t1= ta1.concat()
    t3=ta3.concat()
    #with  tf.control_dependencies([t1 , uinquie_ta2 ,t3  ]):
    final_values = t1 , uinquie_ta2 ,t3 
    return final_values


def multiplyragged_withDesne(tensor1 , dense1) : 
    #this  function multiplies two ragged tesnsors of rank 2 . the most outer ranks of the two tensros must be equal .
    #setting variables and constats 
    outerloop_counter = tf.constant(0 , dtype=tf.int32)
    carry_on = tf.constant(0 , dtype=tf.int32)
    taValues = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL2Splits = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL1Splits = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL1Splits = taL1Splits.write(0,[0]) ## required intialization for L1 split only
    innerloop_processing_graphed = tf.function(innerloop_processing)
    generateL1Tensor_writeback_graphed = tf.function(generateL1Tensor_writeback)
    def outerloop_cond(counter,input1,input2 ,taValues  ,taL2Splits , taL1Splits , carry_on ) :
        with tf.control_dependencies([input1[0],input1[1] , input1[2]]):
            value = tf.shape(input1[2])[0]-1
        return counter < value ## this is the length of the outermost dimision , stop of this 
    def outloop_body(counter,input1,input2, taValues  ,taL2Splits , taL1Splits , carry_on) : 
        l1_comp_begin = input1[2][counter]                  ## this is begin position of the current row in the outer split  ( ie. the ith value in the outer row split tensor ) 
        l1_comp_end = input1[2][counter+1]                  ## this is end position of the current row in the outer split   (ie. the ith + 1 value in the outer row split tensor)

        comp  = innerloop_processing_graphed(l1_comp_begin ,l1_comp_end ,input1  ) ## now retrive the data to be procesed for the selected rows from vector1    
        multiply =tf.matmul(comp , input2) #### This is the desired operation  
        #multiply = tf.Print(multiply,[multiply] , "This is the multipliucation output")
        #input2 = tf.Print(dense1,[dense1] , "This is the variable being learnt  output")
        myshape= tf.shape(multiply) ## calculate the shape of the result in order to prepare to write the result in a ragged tensor format. 
        offset = tf.cond( taValues.size() >0  ,lambda: tf.shape(taValues.concat())[0] , lambda : 0) ### this is a hack, TensorArray.concat returns an error if the array is empty. Thus we check before calling this. 
        #print11=tf.print("=================Final Shape is : " ,myshape[0] , " X " ,myshape[1] )
        l2v = generateL1Tensor_writeback_graphed(offset,myshape[1],myshape[0])  # generate the inner row split of the result for the current element
        taL2Splits=taL2Splits.write(counter,l2v) # write back the inner rowlplit to a TensorArray 
        taValues=taValues.write(counter,tf.reshape(multiply , [-1])) # wirte back the actual ragged tensor elemnts in a another TensorArray
        carry_on=carry_on+myshape[0] ## required to calculate the outer row splite
        taL1Splits=taL1Splits.write(counter+1,[carry_on]) ## This is the outmost row split. 
        with tf.control_dependencies([comp  ,myshape ,l2v , carry_on , multiply ,taValues.concat() , taL2Splits.concat(),taL1Splits.concat()]):
            counter = counter+1
        return counter , input1,input2, taValues  ,taL2Splits , taL1Splits , carry_on
    with tf.name_scope("DenseMultiply") :     
        outerloop_finalcounter , _ , _ , ta1,ta2,ta3,_ = tf.while_loop(outerloop_cond,outloop_body,[outerloop_counter , tensor1 , dense1 ,taValues  ,taL2Splits , taL1Splits,carry_on] , back_prop=True)
    uinquie_ta2 , _ = tf.unique(ta2.concat())  # this is required since some values might be duplicate in the row split itself 
    t1= ta1.concat()
    t3=ta3.concat()
    #with  tf.control_dependencies([t1 , uinquie_ta2 ,t3  ]):
    final_values = t1 , uinquie_ta2 ,t3 
    return final_values

def multiplyraggedTranspose(tensor1) : 
    #this  function multiplies two ragged tesnsors of rank 2 . the most outer ranks of the two tensros must be equal .
    #setting variables and constats 
    outerloop_counter = tf.constant(0 , dtype=tf.int32)
    carry_on = tf.constant(0 , dtype=tf.int32)
    taValues = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL2Splits = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL1Splits = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL1Splits = taL1Splits.write(0,[0]) ## required intialization for L1 split only
    innerloop_processing_graphed = tf.function(innerloop_processing)
    generateL1Tensor_writeback_graphed = tf.function(generateL1Tensor_writeback)
    def outerloop_cond(counter,input1,taValues  ,taL2Splits , taL1Splits , carry_on ) :
        value = tf.shape(input1[2])[0]-1
        return counter < value ## this is the length of the outermost dimision , stop of this 
    def outloop_body(counter,input1,taValues  ,taL2Splits , taL1Splits , carry_on) : 
        l1_comp_begin = input1[2][counter]                  ## this is begin position of the current row in the outer split  ( ie. the ith value in the outer row split tensor ) 
        l1_comp_end = input1[2][counter+1]                  ## this is end position of the current row in the outer split   (ie. the ith + 1 value in the outer row split tensor)
        comp  = innerloop_processing_graphed(l1_comp_begin ,l1_comp_end ,input1  ) ## now retrive the data to be procesed for the selected rows from vector1    
        myop =tf.transpose(comp) #### This is the desired operation  
        
        
        myshape= tf.shape(myop) ## calculate the shape of the result in order to prepare to write the result in a ragged tensor format. 
        offset = tf.cond( taValues.size() >0  ,lambda: tf.shape(taValues.concat())[0] , lambda : 0) ### this is a hack, TensorArray.concat returns an error if the array is empty. Thus we check before calling this. 
        #print11=tf.print("=================Final Shape is : " ,myshape[0] , " X " ,myshape[1] )
        l2v = generateL1Tensor_writeback_graphed(offset,myshape[1],myshape[0])  # generate the inner row split of the result for the current element
        taL2Splits=taL2Splits.write(counter,l2v) # write back the inner rowlplit to a TensorArray 
        taValues=taValues.write(counter,tf.reshape(myop , [-1])) # wirte back the actual ragged tensor elemnts in a another TensorArray
        carry_on=carry_on+myshape[0] ## required to calculate the outer row splite
        taL1Splits=taL1Splits.write(counter+1,[carry_on]) ## This is the outmost row split. 
        with tf.control_dependencies([comp  ,myshape ,l2v , carry_on , myop ]):
            counter = counter+1
        return counter , input1,taValues  ,taL2Splits , taL1Splits , carry_on
    with tf.name_scope("Transponse") :     
        outerloop_finalcounter , _ , ta1,ta2,ta3,_ = tf.while_loop(outerloop_cond,outloop_body,[outerloop_counter , tensor1  ,taValues  ,taL2Splits , taL1Splits,carry_on] , back_prop=True)
    uinquie_ta2 , _ = tf.unique(ta2.concat())  # this is required since some values might be duplicate in the row split itself 
    t1= ta1.concat()
    t3=ta3.concat()
    #with  tf.control_dependencies([t1 , uinquie_ta2 ,t3  ]):
    final_values = t1 , uinquie_ta2 ,t3 
    return final_values



def multiplyraggedDropout(tensor1,rate) : 
    #this  function multiplies two ragged tesnsors of rank 2 . the most outer ranks of the two tensros must be equal .
    #setting variables and constats 
    outerloop_counter = tf.constant(0 , dtype=tf.int32)
    carry_on = tf.constant(0 , dtype=tf.int32)
    taValues = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL2Splits = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL1Splits = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False , infer_shape=False )
    taL1Splits = taL1Splits.write(0,[0]) ## required intialization for L1 split only
    innerloop_processing_graphed = tf.function(innerloop_processing)
    generateL1Tensor_writeback_graphed = tf.function(generateL1Tensor_writeback)
    def outerloop_cond(counter,input1,taValues  ,taL2Splits , taL1Splits , carry_on ) :
        value = tf.shape(input1[2])[0]-1
        return counter < value ## this is the length of the outermost dimision , stop of this 
    def outloop_body(counter,input1,taValues  ,taL2Splits , taL1Splits , carry_on) : 
        l1_comp_begin = input1[2][counter]                  ## this is begin position of the current row in the outer split  ( ie. the ith value in the outer row split tensor ) 
        l1_comp_end = input1[2][counter+1]                  ## this is end position of the current row in the outer split   (ie. the ith + 1 value in the outer row split tensor)
        comp  = innerloop_processing_graphed(l1_comp_begin ,l1_comp_end ,input1  ) ## now retrive the data to be procesed for the selected rows from vector1    
        myop =tf.nn.dropout(comp,rate) #### This is the desired operation  
        
        
        myshape= tf.shape(myop) ## calculate the shape of the result in order to prepare to write the result in a ragged tensor format. 
        offset = tf.cond( taValues.size() >0  ,lambda: tf.shape(taValues.concat())[0] , lambda : 0) ### this is a hack, TensorArray.concat returns an error if the array is empty. Thus we check before calling this. 
        #print11=tf.print("=================Final Shape is : " ,myshape[0] , " X " ,myshape[1] )
        l2v = generateL1Tensor_writeback_graphed(offset,myshape[1],myshape[0])  # generate the inner row split of the result for the current element
        taL2Splits=taL2Splits.write(counter,l2v) # write back the inner rowlplit to a TensorArray 
        taValues=taValues.write(counter,tf.reshape(myop , [-1])) # wirte back the actual ragged tensor elemnts in a another TensorArray
        carry_on=carry_on+myshape[0] ## required to calculate the outer row splite
        taL1Splits=taL1Splits.write(counter+1,[carry_on]) ## This is the outmost row split. 
        with tf.control_dependencies([comp  ,myshape ,l2v , carry_on ,myop]):
            counter = counter+1
        return counter , input1,taValues  ,taL2Splits , taL1Splits , carry_on
    with tf.name_scope("dropout") : 
        outerloop_finalcounter , _ , ta1,ta2,ta3,_ = tf.while_loop(outerloop_cond,outloop_body,[outerloop_counter , tensor1  ,taValues  ,taL2Splits , taL1Splits,carry_on] , back_prop=True)
    uinquie_ta2 , _ = tf.unique(ta2.concat())  # this is required since some values might be duplicate in the row split itself 
    t1= ta1.concat()
    t3=ta3.concat()
    #with  tf.control_dependencies([t1 , uinquie_ta2 ,t3  ]):
    final_values = t1 , uinquie_ta2 ,t3 
    return final_values
