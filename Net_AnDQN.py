import numpy as np
import tensorflow as tf
import TF_utils_cjy as tu

def build(params,net_name):
	master_net = build_local(params,'net_master')
	target_net = build_local(params,'net_target')
	worker_nets = []
	copy_master_to_worker = []
	grad_master = []
	train = []
	global_frame = tf.Variable(0, name='global_step', trainable=False)
	frame_ph = tf.placeholder(tf.int32)
	with tf.device("/cpu:0"): 
		gf_op = tf.assign_add(global_frame,frame_ph)
	lr = tf.placeholder('float')
	rmsprop = tf.train.RMSPropOptimizer(lr,params['rms_decay'],params['rms_momentum'],params['rms_eps'],use_locking=False)


	for i in range(params['num_workers']):
		worker_nets.append(build_local(params,'net_worker_'+str(i),rmsprop))
		copy_op_temp = []
		grad_master_temp = [] 
		for j in range(len(master_net['vars_all'])):
			copy_op_temp.append(tf.assign(worker_nets[i]['vars_all'][j],master_net['vars_all'][j]))
			grad_master_temp.append((worker_nets[i]['grad'][j],master_net['vars_all'][j]))
			#grad_master_temp.append((worker_nets[i]['gvs'][j][0],master_net['vars_all'][j]))

		copy_master_to_worker.append(copy_op_temp)
		grad_master.append(grad_master_temp)
		train.append(rmsprop.apply_gradients(grad_master[-1]))
	copy_master_to_target = []
	for j in range(len(master_net['vars_all'])):
		copy_master_to_target.append(tf.assign(target_net['vars_all'][j],master_net['vars_all'][j]))

	output = {}
	output['master_net'] = master_net
	output['worker_nets'] = worker_nets
	output['copy_ops'] = copy_master_to_worker
	output['train_ops'] = train
	output['target_net'] = target_net
	output['copy_target'] = copy_master_to_target
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

		with tf.variable_scope('fea') as vs_fea:
			#conv_layers
			convs = []
			convs_shapes = []
			inputs = x
			for i in range(len(params['convs_size'])):
				convs.append( tu.conv_layer('conv'+str(i),inputs,params['convs_filter'][i],params['convs_size'][i],params['convs_stride'][i],activation='lrelu') )
				convs_shapes.append( convs[-1].get_shape().as_list() )
				inputs = convs[-1]

			conv_flat,conv_flat_dim = tu.img_to_vec(inputs)			

		#State-Action Value
		with tf.variable_scope('Q') as vs_q:
			fc2 = tu.fc_layer('fc2',conv_flat,hiddens=params['dim_fc'],activation='lrelu')
			if params['LSTM'] : 
				cells = rnn_cell.BasicLSTMCell(params['dim_fc'], forget_bias=1.0,state_is_tuple=True)
				LSTM_h_ph = tf.placeholder('float',[1,params['dim_fc']])  #batch,dim	
				LSTM_c_ph = tf.placeholder('float',[1,params['dim_fc']]) 	
				state_tuple = tf.nn.rnn_cell.LSTMStateTuple(LSTM_h_ph,LSTM_c_ph)	
				conv_flat = tf.reshape(fc2,[1,-1,fc2.get_shape().as_list()[-1]])
				unroll = tf.placeholder(tf.int32,[1])
				fc2, fc2_state = tf.nn.dynamic_rnn(cells,conv_flat,initial_state = state_tuple,sequence_length = unroll)
				fc2 = tf.reshape(fc2,[-1,params['dim_fc']])

			value = tu.fc_layer('value',fc2,hiddens=params['num_actions'],activation='linear')
		
		q_diff = returns-tf.reduce_sum(action*value,1,keep_dims=True)
		loss_q = tf.square(q_diff)
		loss_q = tf.reduce_sum(loss_q)

	#vars
	vars_all = tu.get_all_var_from_net(net_name)

	#grads
	grad = tf.gradients(loss_q,vars_all)
	#gvs = optimizer.compute_gradients(loss_q,vars_all)

	if params['clip_grad']:		
		#gvs = [tf.clip_by_global_norm(grad, params['grad_clip_norm']) for grad in gvs]
		grad = tf.clip_by_global_norm(grad, params['grad_clip_norm'])[0]

	if params['LSTM'] : output['unroll'] = unroll; output['LSTM_h_ph']= LSTM_h_ph ; output['LSTM_c_ph']= LSTM_c_ph ; output['LSTM_state']= fc2_state

	dummy = tf.Variable(0, name='global_step', trainable=False) #for code compatibility with A3C

	output = {'x':x, 
		'action':action,
		'returns':returns,
		'policy':dummy,
		'value':value,
		'loss_total':dummy,
		'vars_all':vars_all,
		'grad' : grad,
		'entropy':dummy,
		'loss_p' : dummy,
		'loss_v' : loss_q,
		'fc2' : fc2
		}

	return output
		
