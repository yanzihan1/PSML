import tensorflow as tf
import numpy as np
import os
from gensim.models.keyedvectors import KeyedVectors


data_path = os.path.abspath('..') + '/data/'
# we use gensim(a useful tool) to get embedding vector, due to the privacy protection,
# we only provide the embedding data
twitter_embedding_path = data_path + 'twitter_embedding.emb'
twitter_vocab_path = data_path + 'twitter_model.vocab'
foursquare_embedding_path = data_path + 'foursquare_embedding.emb'
foursquare_vocab_path = data_path + 'foursquare_model.vocab'
connect_data_path = data_path + 'trainConnect.txt'
connect_test_data_path = data_path + 'testConnect.txt'
# in this simplified  version, we will train our model directly
# connect_warm_up_data_path = data_path + 'trainConnect_400_warm_up.txt'
embedding_size =100
ps_sta=' ' #input_size
# load the embedding vector using gensim
x_vectors = KeyedVectors.load_word2vec_format(foursquare_embedding_path, binary=False, fvocab=foursquare_vocab_path)
y_vectors = KeyedVectors.load_word2vec_format(twitter_embedding_path, binary=False, fvocab=twitter_vocab_path)
inputs = []     # train input vector
labels = []     # train label vector
inputs_neibor=[]
dp_ifps=[]
dp_ifps1=[]
dp_ifps2=[]
test_inputs = []  # test input vectors
test_labels = []  # test label words
test_inputs_neibor=[]
dump_w='dump.datanumber.wk'
dum_file=open('dump_w','w')
x_data_following='../data/foursquare.following'
graph={}
nodes=set()
x_data=open(x_data_following)
for i in x_data:
    edge=i.split()
    nodes.add(edge[0])
    nodes.add(edge[1])
    if graph.get(edge[0]) is None:
        graph[edge[0]] = []
    if graph.get(edge[1]) is None:
        graph[edge[1]] = []
    graph[edge[0]].append(edge[1])
    graph[edge[1]].append(edge[0])

def get_neibor(node):
    neibor=graph.get(node)
    return neibor

def load_data():
    nnn=[]
    f = open(connect_data_path)
    for line in f.readlines():
        line_array = line.strip().split(' ')
        #line_array=[int(x) for x in line_array]
        if line_array[0] not in x_vectors.vocab.keys() or line_array[1] not in y_vectors.vocab.keys():
            print(line_array)
            print("======================warning!!!" + line_array[0] + " or " + line_array[1] + "does not exsits!!!=====================================")
            continue
        inputs.append(x_vectors[line_array[0]])
        labels.append(y_vectors[line_array[1]])
        if int(line_array[0])>ps_sta:
            dp_ifps.append([1.]*embedding_size)
            if int(line_array[0])%2==0:
                #p=np.array([1.]*embedding_size)
                dp_ifps1.append(np.array([1.]*embedding_size))
                dp_ifps2.append(np.array([0.] * embedding_size))
            else:
                dp_ifps1.append(np.array([0.] * embedding_size))
                dp_ifps2.append(np.array([1.] * embedding_size))
        else:
            dp_ifps.append(np.array([0.] * embedding_size))
            dp_ifps1.append(np.array([0.] * embedding_size))
            dp_ifps2.append(np.array([0.] * embedding_size))
        neibor = set(get_neibor(line_array[0]))
        for line in neibor:
            nnn.append(x_vectors[line])
        num = np.array(nnn)
        num_mean = np.mean(num, axis=0)
        inputs_neibor.append(num_mean)
        nnn = []

    print('input size:' + str(len(inputs)))
    print('labels size:' + str(len(labels)))

# this function can be replace by tf.dense in a higher tensorflow version
def add_layer(input_data, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    result = tf.matmul(input_data, weights) + biases
    if activation_function is None:
        outputs = result
    else:
        outputs = activation_function(result)
    return outputs


# record the current data index
data_index = 0


# note that: len(inputs) == len(labels)
def generate_batch(type):
    nnn=[]
    """
    get the batch data
    :param type: train or test
    :return: batch data
    """
    global data_index
    if type == 'train':

        if data_index + batch_size >= len(inputs):  # the case that now_index + batch_size > total data
            batch_inputs = inputs[data_index:]
            batch_labels = labels[data_index:]
            batch_neibor = inputs_neibor[data_index:]

            batch_if1=dp_ifps1[data_index:]
            batch_if2=dp_ifps2[data_index:]
            batch_if=dp_ifps[data_index:]
            data_index = batch_size - len(batch_inputs)
            for d in inputs[:data_index]:
                batch_inputs.append(d)
            for l in labels[:data_index]:
                batch_labels.append(l)
            for b in inputs_neibor[:data_index]:
                batch_neibor.append(b)
            for j in dp_ifps1[:data_index]:
                batch_if1.append(j)
            for k in dp_ifps2[:data_index]:
                batch_if2.append(k)
            for i in dp_ifps[:data_index]:
                batch_if.append(i)
        else:
            batch_inputs = inputs[data_index:data_index + batch_size]
            batch_labels = labels[data_index:data_index + batch_size]
            batch_neibor = inputs_neibor[data_index:data_index + batch_size]
            batch_if2=dp_ifps2[data_index:data_index + batch_size]
            batch_if= dp_ifps[data_index:data_index + batch_size]
            batch_if1 = dp_ifps1[data_index:data_index + batch_size]
            data_index += batch_size
        return batch_inputs, batch_labels,batch_neibor,batch_if1,batch_if2,batch_if
    elif type == 'test':
        f = open(connect_test_data_path)
        for line in f.readlines():
            line_array = line.strip().split(' ')
            if line_array[0] not in x_vectors.vocab.keys() or line_array[1] not in y_vectors.vocab.keys():
                print("======================warning!!!" + line_array[0] + " or " + line_array[
                    1] + "does not exsits!!!=====================================")
                continue
            test_inputs.append(x_vectors[line_array[0]])
            test_labels.append(line_array[1])

        print('test_inputs size:' + str(len(test_inputs)))
        print('test_labels size:' + str(len(test_labels)))
        return test_inputs
    elif type == 'pse':
        test_inputs_neibor = []
        test_if_ps = []
        test_dp_ifps1 = []
        test_dp_ifps2 = []
        f = open(connect_test_data_path)
        for line in f.readlines():
            line_array = line.strip().split(' ')
            if line_array[0] not in x_vectors.vocab.keys() or line_array[1] not in y_vectors.vocab.keys():
                print("======================warning!!!" + line_array[0] + " or " + line_array[
                    1] + "does not exsits!!!=====================================")
                continue
            test_inputs.append(x_vectors[line_array[0]])
            test_labels.append(line_array[1])
            neibor=get_neibor(line_array[0])
            for line in neibor:
                nnn.append(x_vectors[line])
            num=np.array(nnn)
            num_mean=np.mean(num, axis=0)
            test_inputs_neibor.append(num_mean)
            nnn=[]
            #=================================================================
            if int(line_array[0]) > ps_sta:
                test_if_ps.append([1.] * embedding_size)
                if x_vectors[line_array[0]] % 2 == 0:
                    test_dp_ifps1.append(np.array([1.] * embedding_size))
                    test_dp_ifps2.append(np.array([0.] * embedding_size))
                else:
                    test_dp_ifps1.append(np.array([0.] * embedding_size))
                    test_dp_ifps2.append(np.array([1.] * embedding_size))
            else:
                test_if_ps.append(np.array([0.] * embedding_size))
                test_dp_ifps1.append(np.array([0.] * embedding_size))
                test_dp_ifps2.append(np.array([0.] * embedding_size))
        print('test_inputs size:' + str(len(test_inputs)))
        print('test_labels size:' + str(len(test_labels)))
        return test_inputs,test_inputs_neibor,test_if_ps,test_dp_ifps1,test_dp_ifps2


def normalize_vector(vector):
    norm = tf.sqrt(tf.reduce_sum(tf.square(vector), 1, keep_dims=True))
    normalized_embeddings = vector / norm
    return normalized_embeddings


# get the Levenshtein distance
def leven_dis(str1, str2):
    len_str1 = len(str1.lower()) + 1
    len_str2 = len(str2.lower()) + 1
    # create matrix
    matrix = [0 for n in range(len_str1 * len_str2)]
    # init x axis
    for i in range(len_str1):
        matrix[i] = i
    # init y axis
    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]

def add_w_layer(results,emd_size,neibor,ifps,ips0,ips1):
    #note:
           #ipsi: w shape=[[0.]*emd_size] or [[1.]*emd_size]
           #i=0 or i=1
           #this w for update w_1 or w_2.  w_1-->pseudo-anchor_1   w2-->pseudo-anchor_2
           # if pseudo-anchor_0: ips0=[[1.]*emd_size]
           # if pseudo-anchor_1: ips1=[[1.]*emd_size]
           # if pseudo-anchor :ifps:[[1.]*emd_size]
    w0=tf.get_variable('w_0',initializer=[[0.015]*emd_size]*emd_size)
    w1=tf.get_variable('w_1', initializer=[[0.015]*emd_size]*emd_size)

    new_add=tf.matmul(tf.multiply(neibor,ips0),w0)\
           +tf.matmul(tf.multiply(neibor,ips1),w1)
    results=results+tf.multiply(tf.nn.relu(new_add),ifps) #if not,ifps*[0,0,0,0]
    return w0,w1,results
def rank(topn, target):
    result = []
    for item in topn:
        max_length = len(item[0]) if len(item[0]) > len(target) else len(target)
        modify_value = ((max_length / 2.0 - leven_dis(item[0], target) * 1.0) / (max_length / 2.0)) * 0.05
        val = item[1] + modify_value
        if val > 1.0:
            val = 1.0
        if val < 0:
            val = 0
        result.append((item[0], val))
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result


# build net
xs = tf.placeholder(name='xs',dtype=tf.float32, shape=[None, embedding_size])
ys = tf.placeholder(dtype=tf.float32, shape=[None, embedding_size],name='ys')
if_ps= tf.placeholder(dtype=tf.float32, shape= [None, embedding_size],name='if_ps')
if_psu_1= tf.placeholder(dtype=tf.float32,  shape=[None, embedding_size],name='if_psu_1')
if_psu_2= tf.placeholder(dtype=tf.float32,  shape=[None, embedding_size],name='if_psu_2')
neibor=tf.placeholder(dtype=tf.float32, shape=[None,embedding_size],name='neibor')
w1_,w2_,xxs=add_w_layer(xs,embedding_size,neibor,if_ps,if_psu_1,if_psu_2)
#w2_,xxs=add_w2_layer(xs,embedding_size,neibor,ips,i1a2)

hidden_1 = add_layer(xxs, embedding_size, 1200, None)
output_x = add_layer(hidden_1, 1200, embedding_size, None)
results = tf.matmul(normalize_vector(ys), normalize_vector(output_x), transpose_b=True)
loss_x = 1-tf.reduce_mean(tf.diag_part(results))
train_step_x = tf.train.GradientDescentOptimizer(1).minimize(loss_x)




init = tf.global_variables_initializer()
num_steps = 60001
# due to the data is not big, we set a small batch size
batch_size = 1

with tf.Session() as session:
    print("program begin")
    init.run()
    load_data()
    #batch_size = 1
    average_loss = 0
    pse_loss = 0
    for step in range(num_steps):
        if step==320:
            a=3
        batch_inputs, batch_labels,batch_neibor,batch_ifps1,batch_ifps2,batch_ifps=generate_batch('train')
        feed_dict = {xs: batch_inputs, ys: batch_labels,if_ps:batch_ifps,if_psu_1:batch_ifps1,if_psu_2:batch_ifps2,neibor:batch_neibor}
        loss_val, _,outputx = session.run([loss_x, train_step_x,output_x], feed_dict=feed_dict)
        average_loss += loss_val

        #
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
                pse_loss/=2000
            print("Average loss_x at step ", step, ": ", average_loss)
            #print("Average loss_x_2 at step ", step, ": ", pse_loss)
            average_loss = 0
            pse_loss = 0
        if step % 20000 == 0 and step>0:
            test_inputs = []
            test_labels = []
            test_input,test_neibor,test_if,test_if1,test_if2=generate_batch('pse')
            test_feed={xs:test_input,if_ps:test_if,if_psu_1:test_if1,if_psu_2:test_if2,neibor:test_neibor}
            prediction = session.run(output_x, feed_dict=test_feed)
            #prediction = session.run(output_x_2, feed_dict={f_results: prediction1, neibor: generate_batch('pse')})
            count = 0
            total = np.zeros(101)
            for vector in prediction:
                number_in_topn = 0
                topn = y_vectors.similar_by_vector(vector=vector, topn=100)
                rank_result = rank(topn, test_labels[count])
                for item in rank_result:
                    number_in_topn += 1
                    if item[0] == test_labels[count]:
                        index = number_in_topn
                        while index < 101:
                            total[index] += 1
                            index += 1
                count += 1
            score=[]
            for i in range(1,31):
                score.append(total[i] / count)
                #if i in [1, 5, 10, 15, 30,50]:
                #    print('top ' + str(i) + ' : ' + str(total[i] / count))
            print("score",score)
