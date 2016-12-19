'''
Networks for Asynchronous Methods for Deep Reinforcement Learning 
Jinyoung Choi
'''
import numpy as np
import tensorflow as tf
import TF_utils_cjy as tu
from tensorflow.python.ops import rnn, rnn_cell
def build(params,net_name):
	master_net = build_local(params,'net_master')
	worker_nets = []
	copy_op = []
	grad_master = []
	train = []
	#global_step = tf.Variable(0, name='global_step', trainable=False)
	global_frame = tf.Variable(0, name='global_step', trainable=False)
	frame_ph = tf.placeholder(tf.int32)
	with tf.device("/cpu:0"): 
		gf_op = tf.assign_add(global_frame,frame_ph)
	lr = tf.placeholder('float')
	rmsprop = tf.train.RMSPropOptimizer(lr,params['rms_decay'],params['rms_momentum'],params['rms_eps'],use_locking=False)

	for i in range(params['num_workers']):
		worker_nets.append(build_local(params,'net_worker_'+str(i)))
		copy_op_temp = []
		grad_master_temp = [] 
		for j in range(len(master_net['vars_all'])):
			copy_op_temp.append(tf.assign(worker_nets[i]['vars_all'][j],master_net['vars_all'][j]))
			grad_master_temp.append((worker_nets[i]['grad'][j],master_net['vars_all'][j]))

		copy_op.append(copy_op_temp)
		grad_master.append(grad_master_temp)
		train.append(rmsprop.apply_gradients(grad_master[i]))

	output = {}
	output['master_net'] = master_net
	output['worker_nets'] = worker_nets
	output['copy_ops'] = copy_op
	output['train_ops'] = train
	#output['global_step_rms'] = global_step
	output['global_frame'] = global_frame
	output['global_frame_ph'] = frame_ph
	output['global_frame_op'] = gf_op
	output['lr_ph'] = lr

	return output
	

def build_local(params,net_name,device="/gpu:0"):
	print 'Building ' + net_name
	with tf.variable_scope(net_name) as vs:	
		#input
		x = tf.placeholder('float',[None,params['img_h'],params['img_w'],params['img_c']*params['history']],name='x')
		action = tf.placeholder("float", [None, params['num_actions']],name='actions')
		returns = tf.placeholder("float",[None,1],name='returns')

		#conv_layers
		with tf.variable_scope('fea') as vs_fea:
			convs = []
			convs_shapes = []
			inputs = x
			for i in range(len(params['convs_size'])):
				convs.append( tu.conv_layer('conv'+str(i),inputs,params['convs_filter'][i],params['convs_size'][i],params['convs_stride'][i],activation='relu') )
				convs_shapes.append( convs[-1].get_shape().as_list() )
				inputs = convs[-1]

			conv_flat,conv_flat_dim = tu.img_to_vec(inputs)		
	
		#common fc/lstm
		with tf.variable_scope('common') as vs_c:
			fc2 = tu.fc_layer('fc2',conv_flat,hiddens=params['dim_fc'],activation='relu')
			if params['LSTM'] : 
				cells = rnn_cell.BasicLSTMCell(params['dim_fc'], forget_bias=1.0,state_is_tuple=True)
				LSTM_h_ph = tf.placeholder('float',[1,params['dim_fc']])  #batch,dim	
				LSTM_c_ph = tf.placeholder('float',[1,params['dim_fc']]) 	
				state_tuple = tf.nn.rnn_cell.LSTMStateTuple(LSTM_c_ph,LSTM_h_ph)	
				fc2 = tf.reshape(fc2,[1,-1,fc2.get_shape().as_list()[-1]])
				unroll = tf.placeholder(tf.int32,[1])
				fc2, fc2_state = tf.nn.dynamic_rnn(cells,fc2,initial_state = state_tuple,sequence_length = unroll)
				fc2 = tf.reshape(fc2,[-1,params['dim_fc']])

		#State Value
		with tf.variable_scope('v') as vs_v:
			value = tu.fc_layer('value',fc2,hiddens=1,activation='linear')
		#Softmax_Policy
		with tf.variable_scope('p') as vs_p:
			policy = tu.fc_layer('policy',fc2,hiddens=params['num_actions'],activation='softmax')

		#Loss
		log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
		advantage = returns - value
		loss_ac_p = tf.reduce_sum(action*log_policy,1,keep_dims=True)*tf.stop_gradient(advantage)
		loss_ac_v = 0.5*tf.nn.l2_loss(advantage)

		entropy = -tf.reduce_sum(log_policy*policy,1,keep_dims=True)
		loss_ac_p += params['entropy_reg_coeff']*entropy		

		loss_ac_p = -tf.reduce_sum(loss_ac_p) #Policy gradient uses gradient ascent (not descent!)
		loss_ac_v = tf.reduce_sum(loss_ac_v)

		loss_total = loss_ac_p+loss_ac_v
	

	#grads
	vars_all = tu.get_all_var_from_net(net_name)	
	gvs = tf.gradients(loss_total,vars_all)


	if params['clip_grad']:		
		gvs = [tf.clip_by_norm(grad, params['grad_clip_norm']) for grad in gvs]
		#gvs = tf.clip_by_global_norm(gvs, params['grad_clip_norm'])[0]

	#print gvs
	output = {'x':x, 
		'action':action,
		'returns':returns,
		'policy':policy,
		'value':value,
		'loss_total':loss_total,
		'vars_all':vars_all,
		'grad' : gvs,
		'entropy':tf.reduce_sum(entropy),
		'loss_p' : loss_ac_p,
		'loss_v' : loss_ac_v,
		'fc2' : fc2
		}

	if params['LSTM'] : output['unroll'] = unroll; output['LSTM_h_ph']= LSTM_h_ph ; output['LSTM_c_ph']= LSTM_c_ph ; output['LSTM_state']= fc2_state

	return output
		
