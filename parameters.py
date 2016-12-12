'''
Hyperparameters for Asynchronous Methods for Deep Reinforcement Learning 
Jinyoung Choi
'''
def load_params():
	params = {
		#Meta
		'log_name':'A3C',
		'log_path':'logs/',
		'ckpt_path':'ckpt/',
		'ckpt_file' : None,
		'eval_mode':False,
		'eval_wait':0., #seconds
		#environment
		'rom' : 'pong', #without .bin
		'show_0th_thread' : True,
		'show_eval' : True,
		'frameskip' : 4,
		'num_actions' : 0,
		'img_w' : 84,
		'img_h' : 84,
		'img_c' : 1,
		'history' : 4,
		'repeat_prob':0.,
		#Networks
		'net_type' : 'A3C',
		'num_workers': 16,
		'convs_size': [8,4],
		'convs_filter' : [16,32],
		'convs_stride' : [4,2],
		'dim_fc' : 256, 
		'entropy_reg_coeff':0.01,
		'max_step' : 5, #unroll in LSTM
		'LSTM' : True,
		#training
		'discount' : 0.99,
		'lr_init' : 0.0007,
		'lr_loguniform_low':1e-4,
		'lr_loguniform_high':1e-2,
		'lr_loguniform_seed':0.4226,
		'lr_zero_frame' : 30*4*(10**6),
		'rms_decay':0.99,
		'rms_momentum':0.0,
		'rms_eps':0.1,
		'max_T' : 30*4*(10**6),
		'score_display_interval' : 5, #episodes
		'save_interval' : 1000000, #frames
		'eval_interval' : 4*(10**6), #frames
		'eval_duration' : 10,#episodes
		'clip_grad' : True,
		'grad_clip_norm' : 40.,
		'clip_reward' : True,
		'eps_max' : [1.0,1.0,1.0],
		'eps_min' : [0.1,0.01,0.5],
		'eps_frame' : [1000000,1000000,1000000],
		'eps_prob' : [0.4,0.3,0.3],
		'target_copy_interval' : 30000,
		}

	if params['rom'] == 'toy_way':
		params['convs_size'] = []
		params['convs_filter'] = []
		params['convs_stride'] = []
		params['field_size'] = 4
		params['pixel_per_grid'] = 5
		params['history'] = 1
		params['num_waypoints'] = 1
		params['reward_move'] = -0.1
		params['reward_waypoint'] = 1.0
		params['reward_clear'] = 0.0
		params['reward_timeout'] = 0
		params['timeout'] = 100
		params['frameskip'] = 1
		params['img_h'] = params['field_size'] * params['pixel_per_grid']
		params['img_w'] = params['field_size'] * params['pixel_per_grid']
		params['lr_init'] = 0.0001
		params['eval_interval'] = 10000 #frames
		params['eval_duration'] = 10#episodes

	return params
