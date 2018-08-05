
# coding: utf-8

# In[1]:


import tensorflow as tf
import sys
sys.path.append("../transformer/")
from data_load import get_batch_data, load_de_vocab, load_en_vocab


# In[2]:


class Hyperparams(object):
    #data
    source_train = 'corpora/train.tags.de-en.de'
    target_train = 'corpora/train.tags.de-en.en'
    source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    
    # training
    batch_size=128
    lr = 0.001
    
    #model
    maxlen = 10 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    units_per_head=1024
    dropout_rate = 0.1
    is_training=True
    num_layers=6
hp=Hyperparams()


# In[3]:


def multihead_attention(query,key,value,seq_len,units_per_head=1024,num_heads=8,mask=False,scope="multihead_attention", 
reuse=None):
    #使用时一般K和V一样
    #query(N,Tq/h,C) key=value(N,Tk/h,C)
    with tf.variable_scope(scope, reuse=reuse):
#         units_per_head = query.shape[-1]
        num_units = query.shape[-1]
        Q=tf.layers.dense(query,num_units,activation=tf.nn.relu)# (N, T_q, C)
        K=tf.layers.dense(key,num_units,activation=tf.nn.relu)# (N, T_k, C)
        V=tf.layers.dense(value,num_units,activation=tf.nn.relu)# (N, T_k, C)
        #tf.split(dimension, num_split, input)：dimension的意思就是输入张量的哪一个维度，
        #如果是0就表示对第0维度进行切割。num_split就是切割的数量
        Q=tf.concat(tf.split(Q,num_heads,axis=2),axis=0)#(h*N,Tq,C/h)
        K=tf.concat(tf.split(K,num_heads,axis=2),axis=0)#(h*N,Tk,C/h)
        V=tf.concat(tf.split(V,num_heads,axis=2),axis=0)#(h*N,Tk,C/h)
        #计算内积，然后mask，然后softmax
        A = tf.matmul(Q,tf.transpose(K,[0,2,1]))#(h*N,Tq,Tk)
        A = A /(K.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        #将元素中最后一个为度之和全为0的位置标记为0 source(N, T_k,word_dim)
        key_masks = tf.sign(tf.abs(tf.reduce_sum(key, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        #由于每个queries都要对应这些keys，而mask的key对每个queries都是mask的
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, Q.shape[1], 1]) # (h*N, T_q, T_k)
        #定义一个和outputs同shape的paddings，该tensor每个值都设定的极小。用where函数比较，当对应位置的key_masks值为0
        #也就是需要mask时，outputs的该值（attention score）设置为极小的值（利用paddings实现），否则保留原来的outputs值。
        paddings = tf.ones_like(A)*(-2**32+1)
        A = tf.where(tf.equal(key_masks, 0), paddings, A) # (h*N, T_q, T_k)

        #Causality Future blinding
        if mask:
            #首先定义一个和outputs后两维的shape相同shape（T_q,T_k）的一个张量（矩阵）。
            #然后将该矩阵转为三角阵tril。三角阵中，对于每一个T_q,凡是那些大于它角标的T_k值全都为0，
            #这样作为mask就可以让query只取它之前的key（self attention中query即key）。
            #由于该规律适用于所有query，接下来仍用tile扩展堆叠其第一个维度，构成masks，shape为(h*N, T_q,T_k).
            diag_vals = tf.ones_like(A[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(A)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, A) # (h*N, T_q, T_k)
        A = tf.nn.softmax(A)# (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(query, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(key)[1]]) # (h*N, T_q, T_k)

        A *= query_masks # broadcasting. (N, T_q, C)
        A=tf.layers.dropout(A, rate=hp.dropout_rate)

        # weighted sum
        outputs=tf.matmul(A,V)# ( h*N, T_q, C/h)

        #Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        # Residual connection
        outputs += query

        #Normalize
        outputs=layer_normal(outputs)
    
    return outputs
def feedforward(inputs,num_units=[2048,512],reuse=None):
    outputs= tf.layers.conv1d(
        inputs,filters=num_units[0],kernel_size=1,activation=tf.nn.relu,use_bias=True)
    outputs=tf.layers.conv1d(
        outputs,filters=num_units[1],kernel_size=1,activation=None,use_bias=True)
    # Residual connection
    outputs += inputs
    outputs=layer_normal(outputs)
    return outputs
def layer_normal(inputs,epsilon = 1e-8):
    mean,variance = tf.nn.moments(inputs,[-1],keep_dims=True)
    params_shape = inputs.shape[-1:]
    beta= tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs-mean)/(((variance + epsilon) ** (.5)))
    outputs = gamma * normalized + beta
    return outputs
def encoder (model,num_layers,hparams,scope,seq_len):
    for i in range(num_layers):
        scope="encoder_layer_{}".format(i)
        with tf.variable_scope(scope):
            model.enc = multihead_attention(
                query=model.enc,
                key=model.enc,
                value=model.enc,
                units_per_head=hparams.units_per_head,
                num_heads=hparams.num_heads,
                mask=False,
                seq_len=seq_len)
            model.enc = feedforward(model.enc, num_units=[4*hparams.hidden_units, hparams.hidden_units])
def decoder (model,num_layers,hparams,scope,seq_len):
    for i in range(num_layers):
        scope="decoder_layer_{}".format(i)
        with tf.variable_scope(scope):
            model.dec=multihead_attention(
                query=model.dec,
                key=model.dec,
                value=model.dec,
                units_per_head=hparams.units_per_head,
                num_heads=hparams.num_heads,
                mask=True,
                scope="self_attention",
                seq_len=seq_len)
            model.dec=multihead_attention(
                query=model.dec,
                key=model.enc,
                value=model.enc,
                units_per_head=hparams.units_per_head,
                num_heads=hparams.num_heads,
                mask=False,
                scope="vanilla_attention",
                seq_len=seq_len)
            model.dec = feedforward(model.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
def position_embedding(inputs,position_size):
    batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
    pos_j = 1. / tf.pow(10000., 2 * tf.range(position_size / 2, dtype=tf.float32) / position_size)
    pos_j = tf.expand_dims(pos_j, 0)
    pos_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    pos_i = tf.expand_dims(pos_i, 1)
    pos_ij = tf.matmul(pos_i, pos_j)
    pos_ij = tf.concat([tf.cos(pos_ij), tf.sin(pos_ij)], 1)
    position_embedding = tf.expand_dims(pos_ij, 0) + tf.zeros((batch_size, seq_len, position_size))
    return position_embedding
def word_embedding(inputs,vocab_size,num_units,zero_pad=True,scale=True,scope="embedding",reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs = outputs * (num_units ** 0.5) 
    return outputs

def label_smoothing(inputs,epsilon=0.1):
    """Applies label smoothing. See https://arxiv.org/abs/1512.00567."""
    K = tf.to_float(tf.shape(inputs)[-1]) # number of channels
    return ((1.-epsilon)*inputs+(epsilon/K))
#     return ((1-epsilon)*inputs+(epsilon/K))


# In[4]:


class transformer(object):
    def __init__(self):
        if hp.is_training:
            self.x,self.y,num_batch = get_batch_data() # (N, T)
        else: #inference
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

        

        # Load vocabulary    
        de2idx, idx2de = load_de_vocab()
        en2idx, idx2en = load_en_vocab()
        #[N,T,hp.hidden_units]
        self.enc = word_embedding(self.x,vocab_size=len(de2idx),num_units=hp.hidden_units,scale=True,scope="enc_embed")
        #[N,T,hp.hidden_units]??
        self.enc+= position_embedding(self.x,hp.hidden_units)
        self.enc = tf.layers.dropout(self.enc, rate=hp.dropout_rate,training=tf.convert_to_tensor(hp.is_training))
        seq_len=hp.maxlen
        encoder(self,hp.num_layers,hp,"encoder",seq_len)
        
        self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>
        self.dec = word_embedding(self.decoder_inputs, vocab_size=len(en2idx), 
                                  num_units=hp.hidden_units,scale=True,scope="dec_embed")
        self.dec+=position_embedding(self.x,hp.hidden_units)
        self.dec = tf.layers.dropout(self.dec, rate=hp.dropout_rate,training=tf.convert_to_tensor(hp.is_training))
        
        decoder(self,hp.num_layers,hp,"decoder",seq_len)
        
        
        # Final linear projection
        self.logits = tf.layers.dense(self.dec, len(en2idx)) #[N,T,len(en2idx)]
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1)) #[N,T]
        #同时把label（即self.y）中所有id不为0（即是真实的word，不是pad）的位置的值用float型的1.0代替作为self.istarget，其shape为[N,T]
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        #当self.preds和self.y中对应位置值相等时转为float 1.0,否则为0
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
        
        if hp.is_training:
            #Loss
            
            self.y_smoothed = label_smoothing(tf.cast(tf.one_hot(self.y,depth=len(en2idx)),dtype=tf.float32))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.y_smoothed)
            self.mean_loss=tf.reduce_sum(self.loss*self.istarget)/(tf.reduce_sum(self.istarget))
            
            #Training Scheme
            self.global_step=tf.Variable(0,name='global_step',trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr,beta1=0.9,beta2=0.98,epsilon=1e-8)
            self.train_op= self.optimizer.minimize(self.loss,global_step=self.global_step)
            
            #Summary
            tf.summary.scalar('mean_loss',self.mean_loss)
            self.merged = tf.summary.merge_all()


# In[ ]:


def train():
    # Load vocabulary    
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    # Start session
    model=transformer()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
#         for epoch in range(1, hp.num_epochs+1):
        sess.run(model.train_op)
        


# In[ ]:


train()

