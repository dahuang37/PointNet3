import os
import torch
import time
from datetime import datetime
from .logger import Logger

class Saver(object):

	def __init__(self, opt):
		self.debug = opt.debug

		if self.debug == 0:
			self.save_path = datetime.now().strftime('%Y-%m-%d-%H:%M')
			self.save_path = "checkpoints/" + opt.outf + "/" + opt.model + "/" + self.save_path
			self.train_path = self.save_path + "/train"
			self.test_path = self.save_path + "/test"

			try:
			    os.makedirs('%s' % self.train_path)
			    os.makedirs('%s' % self.save_path)
			except OSError:
			    pass

			os.system('cp -r models %s' % (self.save_path)) # bkp of model def
			os.system('cp train.py %s' % (self.save_path))
			os.system('cp -r utils %s' % (self.save_path))

			self.log_file = open(self.save_path+"/log_train.txt", 'w')
			# self.log_file.write(str(opt)+'\n')

			# set up logger for loss, accuracy graph
			self.train_logger = Logger(self.train_path)
			self.test_logger = Logger(self.test_path)

		self.starting_time = time.time()
		self.best_accuracy = 0
		self.best_avg_accuracy = 0
		self.best_model_wts = None
		
		self.log_string(str(opt)+'\n')

	def log_string(self, out_str):
	    if self.debug==0:
	        self.log_file.write(out_str+'\n')
	        self.log_file.flush()
	    print(out_str)

	def save_result(self):
		# record time
		time_elapsed = time.time() - self.starting_time
		print_time = 'Training complete in {:.0f}m {:.0f}s'.format(
		    time_elapsed // 60, time_elapsed % 60)
		self.log_string(print_time)

		self.log_string('Best val Acc: {:2f}'.format(100.*self.best_accuracy))
		self.log_string('Best val Avg Acc: {:2f}'.format(100.*self.best_avg_accuracy))

		if self.debug==0:
			if self.best_model_wts is not None:
				torch.save(self.best_model_wts, '%s/model.pth' % (self.save_path))

	def log_parameters(self, parameters):
		"""
		calculates the number of parameters in the model
		"""
		total_parameters = 0
		for parameter in parameters:
		    # self.log_string(str(parameter.size()))
		    temp = 1
		    for i in range(len(parameter.size())):
		        temp *= parameter.size()[i]
		    total_parameters += temp
		self.log_string("Total parameters: %d" % total_parameters)

	def update_training_info(self, train, loss, accuracy, avg_accuracy, model_wts, step):
		if not train:
			if accuracy > self.best_accuracy:
				self.best_accuracy = accuracy
				self.best_model_wts = model_wts
			if avg_accuracy > self.best_avg_accuracy:
				self.best_avg_accuracy = avg_accuracy

		if(self.debug == 0):
			info = {
			    'loss': loss,
			    'accuracy': accuracy
			}

			for tag, value in info.items():
				if train:
					self.train_logger.scalar_summary(tag, value, step)
				else:
					self.test_logger.scalar_summary(tag, value, step)




	