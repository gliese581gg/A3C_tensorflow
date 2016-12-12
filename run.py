'''
Asynchronous Methods for Deep Reinforcement Learning 
Jinyoung Choi
'''
import numpy as np
import cv2
import tensorflow as tf
import time
import TF_utils_cjy as tu
import sys
import argparse
import env_way
import thread
import worker
import parameters
import env_atari

params = parameters.load_params()

ap = argparse.ArgumentParser()
ap.add_argument("-log", "--log_name", required = False, help = "log file name")
ap.add_argument("-net", "--net_type", required = False, help = "network type('A3C' or 'AnDQN')")
ap.add_argument("-LSTM", "--LSTM", required = False, help = "LSTM (True or False)")
ap.add_argument("-show_eval", "--show_eval", required = False, help = "show evaluation screen? (True or False)")
ap.add_argument("-eval_mode", "--eval_mode", required = False, help = "Evaluation only (True or False)")
ap.add_argument("-ckpt", "--ckpt_name", required = False, help = "checkpoint name (without path)")
ap.add_argument("-rom", "--rom", required = False, help = "game rom name without '.bin' ('toy_way' for toy problem)")
args = vars(ap.parse_args())
print args
for i in args.keys():
	if i in params.keys() and args[i] is not None:
		if args[i] == 'True' : aar = True
		elif args[i] == 'False' : aar = False
		else : aar = args[i]
		params[i] = aar

if params['eval_mode'] : params['num_workers'] = 0
#if params['LSTM'] : params['history'] = 1


#environment
if params['rom'] == 'toy_way':	env = env_way.env_way(params)
else : 
	env = env_atari.env_atari(params)
	#log-uniform learning rate setting (reference : https://github.com/miyosuda/async_deep_reinforce )
	params['lr_init'] = np.exp(np.log(params['lr_loguniform_low']) * (1-params['lr_loguniform_seed']) + np.log(params['lr_loguniform_high']) * params['lr_loguniform_seed'])
img = env.reset()
params['num_actions'] = env.action_space.n


if params['show_eval'] : 
	cv2.startWindowThread()
	cv2.namedWindow('Evaluation')

#build_networks
if params['net_type'] == 'A3C':
	import Net_A3C
	net = Net_A3C.build(params,'A3C')
elif params['net_type'] == 'AnDQN':
	import Net_AnDQN
	net = Net_AnDQN.build(params,'AnDQN')
else : raise ValueError
workers = []

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

eval_var = tf.Variable(0,name='eval_var')
eval_ph = tf.placeholder(tf.int32)
with tf.device("/cpu:0"): eval_switch = tf.assign(eval_var,eval_ph,name='eval_switch_op')

saver = tf.train.Saver() #TODO save only the master net
sess = tf.Session(config=gpu_config)
sess.run(tf.initialize_all_variables())


summary_op = tf.merge_all_summaries()
log = params['log_path']+params['log_name']
summary_writer = tf.train.SummaryWriter(log, sess.graph_def)

worker_summary_dict = {'op':summary_op,'writer':summary_writer}


if params['ckpt_file'] is not None : 
	print 'Continue from ',params['ckpt_file']
	saver.restore(params['ckpt_file'])

#Create workers
for i in range(params['num_workers']):
		print 'Initializing Thread ' + str(i)
		# worker_idx,params,worker_net,copy_master_to_worker,copy_master_to_target,train_p,train_v,session,master,target,global_step
		workers.append(worker.worker(i,params,net,sess,eval_var,worker_summary_dict))
		thread.start_new_thread(workers[i].run_worker,(i,))

#Start training
gf = sess.run(net['global_frame'])
last_eval_frame = gf
last_save = gf
last_target_copy = gf
sess.run(eval_switch,{eval_ph:0})

print 'Start learning. Hyper paramters are:'
print params

def preprocess(imag):
	result = imag.copy()
	if params['rom'] != 'toy_way' and (np.array(result.shape)!=np.array([params['img_h'],params['img_w'],params['img_c']])).any():
		result = cv2.resize(result,(84,110))
		result = result[18:102,:,:]

	if params['img_c'] == 1 :  result =  cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
	result = result.reshape((params['img_h'],params['img_w'],params['img_c']))
	return result

while gf < params['max_T'] :

	for ii in range(len(workers)):
		if workers[ii].dead : raise ValueError

	gf = sess.run(net['global_frame'])

	if params['net_type'] == 'AnDQN':
		if gf > last_target_copy + params['target_copy_interval'] : 
			sess.run(net['copy_target'])
			last_target_copy = gf

	if gf > last_save + params['save_interval'] : 
		saver.save(sess, params['ckpt_path']+'ckpt_'+str(gf))
		print 'Model saved as ckpt/ckpt'+str(gf)
		last_save = gf

	if gf > last_eval_frame + params['eval_interval'] or params['eval_mode']:
		sess.run(eval_switch,{eval_ph:1})
		print 'Start Evaluation! (Training is stopped)'
		epi_reward = 0.
		acc_reward=0.
		num_epi = 0
		if params['LSTM'] : LSTM_h = np.zeros((1, params['dim_fc'])) ; LSTM_c = np.zeros((1, params['dim_fc']))
		img = env.reset()
		per = np.zeros((1,params['img_h'],params['img_w'],params['img_c']*params['history']))	
		eval_start_time = time.time()
		epi_end = 0
		while num_epi < params['eval_duration']:
			per[0,:,:,0:params['img_c']*(params['history']-1)] = per[0,:,:,params['img_c']:params['img_c']*params['history']].copy()
			per[0,:,:,params['img_c']*(params['history']-1):] = preprocess(img)/255.0

			fd = {}
			fd[net['master_net']['x']]=per

			if params['LSTM'] : 
				fd[net['master_net']['LSTM_h_ph']] = LSTM_h ; fd[net['master_net']['LSTM_c_ph']] = LSTM_c ; fd[net['master_net']['unroll']] = np.array([1])
				pol,val,LSTM_c_h_temp = sess.run([net['master_net']['policy'],net['master_net']['value'],net['master_net']['LSTM_state']],feed_dict = fd)
				LSTM_c = LSTM_c_h_temp[0] 
				LSTM_h = LSTM_c_h_temp[1] 
			else :
				pol,val = sess.run([net['master_net']['policy'],net['master_net']['value']],feed_dict = fd)
			pol=pol.reshape(-1);val=val.reshape(-1)
	
			if params['net_type'] == 'A3C':
				action = params['num_actions']-1
				seed = np.random.random()
				acc_prob = 0.

				for i in range(params['num_actions']):
					acc_prob += pol[i]
					if seed < acc_prob : action = i ; break

			elif params['net_type'] == 'AnDQN':				
				action = np.argmax(val)

			step_reward = 0
			epi_end = 0

			if params['show_eval'] : cv2.imshow('Evaluation',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

			img,step_reward,epi_end,info=env.step(action)
				
			epi_reward += step_reward

			if epi_end == 1:
				img = env.reset()
				per = np.zeros((1,params['img_h'],params['img_w'],params['img_c']*params['history']))	
				LSTM_h = np.zeros((1, params['dim_fc']))
				LSTM_c = np.zeros((1, params['dim_fc']))

				print '    eval_episode_'+str(num_epi)+' score : ' + str(epi_reward)

				acc_reward += epi_reward
				epi_end = 0
				epi_reward = 0
				num_epi += 1

			time.sleep(params['eval_wait'])


		print 'Evaluation Running Time : ' + str(time.time()-eval_start_time) + ' (# of frames learned: ' + str(gf) +' / ' + str(params['max_T'])+')'
		print '    average_reward : ' + str(acc_reward/max(1,num_epi)) + ' (' + str(num_epi) + ' episodes)'
		print 'Continue learning!'
		summary_data = tf.Summary()
		summary_data.value.add(tag='Evaluation_mean_score', simple_value=float(acc_reward/max(1,num_epi)))
		summary_writer.add_summary(summary_data, gf)
		summary_writer.flush()
		last_eval_frame = gf
		sess.run(eval_switch,{eval_ph:0})




