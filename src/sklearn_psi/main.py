'''
Created on 22/10/2012

@author: jmontgomery
'''

import web
import json
import uuid
import cPickle
import os
import sys
from sklearn.linear_model import LogisticRegression

urls = (
	'/', 'index',
	'/learner/ranking', 'rank',
	'/learner/(\w+).(\w+)', 'train',
	'/predictor/([-\w]+)', 'basic',
	'/predictor/([-\w]+)/update', 'update'
)

wrong_feature_count = { 'badRequest' : 'Wrong number of features' }

class PredictorModel:
	def __init__(self, pred, featureCount, labels=None):
		self.name = str(uuid.uuid4())
		self.pred = pred
		self.featureCount = featureCount
		self.labels = labels
		self.predictProb = False
		
	def save(self):
		return self.replace(self.pred)
	
	def replace(self, pred):
		self.pred = pred
		out = open(PredictorModel.path(self.name), 'wb')
		cPickle.dump(self, out, 2)
		out.close()
		return self
	
	def hasLabels(self):
		return not self.labels is None
	
	def predict(self, instances):
		#Currently 'predict probability' is interpreted as meaning there were only two training 'classes', and 2nd's probability will be reported 
		if self.predictProb:
			values = [ v[1] for v in self.pred.predict_proba(instances) ]
		else:
			values = self.pred.predict(instances).tolist()
		if self.hasLabels():
			values = [ self.labels[int(y)] for y in values ]
		return values

	@classmethod
	def dir(cls):
		return os.environ['SKLEARN_PREDICTOR_DIR'] if os.environ.has_key('SKLEARN_PREDICTOR_DIR') else '.'

	@classmethod
	def path(cls, name):
		return os.path.join(PredictorModel.dir(), name + '.bin')

	@classmethod
	def load(cls, name):
		try:
			predFile = open(PredictorModel.path(name), 'rb')
			model = cPickle.load(predFile)
			predFile.close()
			return model
		except IOError:
			return None

	@classmethod
	def delete(cls, name):
		os.remove(PredictorModel.path(name))
		
	def labelsToInt(self, values):
		if not self.hasLabels():
			return values
		targetMap = dict( [ (y, i) for i, y in enumerate( self.labels ) ] )
		return [ targetMap.get(y) for y in values ]
	
	
class index:
	def GET(self):
		return 'scikit-learn PSI wrapper service is running\n'
		
class rank:
	def POST(self):
		params, data = train.split_params(web.data())
		#Formulate data['source'] and data['target'] in some sensible way; this way could work, but is not sensible
		source = data['preferred'] + data['not_preferred']
		target = [1] * len(data['preferred']) + [-1] * len(data['not_preferred'])  
		model, c = train.train(LogisticRegression, params, { 'source' : source, 'target' : target})
		model.predictProb = True
		return train.persist_model(model, c)

class train:
	def POST(self, moduleName, learnerName):
		# Split parameters from training data (currently receive *output* from source and target attributes, not their references
		params, data = train.split_params(web.data())
		# Replace any unicode values with str; what a pain
		for key, value in params.items():
			if isinstance(value, unicode):
				params[key] = str(value)
		# Load learner/predictor using given learnerID (really Python package.class name)
		module = __import__('sklearn.' + moduleName, fromlist=['sklearn'])
		model, clusters = train.train(getattr(module, learnerName), params, data)
		return train.persist_model(model, clusters)
	
	@classmethod
	def split_params(cls, web_data):
		params = json.loads(web_data)
		data = params['resources']
		del params['resources']
		return params, data
	
	@classmethod
	def train(cls, learnerClazz, params, data):
		''' Trains the predictor and returns a PredictorModel wrapper and, if predictor is clusterer, the number of clusters (None if it is not) '''
		pred = learnerClazz(**params)
		clusters = None
		model = PredictorModel(pred, len(data['source'][0]), data.get('targetLabels'))
		if data.has_key('target'):
			pred.fit(data['source'], model.labelsToInt(data['target']) if model.hasLabels() else data['target'])
		else:
			pred.fit( data['source'] )
			#An incomplete list of the ways to discover, after training, how many clusters a scikit-learn clustering algorithm has identified
			if hasattr(pred, 'predict_proba'):
				clusters = len( pred.predict_proba( [ data['source'][0] ] )[0] )
			elif hasattr(pred, 'cluster_centers_'):
				clusters = len(pred.cluster_centers_) 
		return model, clusters
	
	@classmethod
	def persist_model(cls, model, clusters):
		model.save()
		response = {'predictor' : model.name}
		if clusters:
			response['clusters'] = clusters
		return json.dumps(response)

class basic:
	def GET(self, name):
		model = PredictorModel.load(name)
		if model is None: return web.notfound()
		return 'Found this predictor: ' + str(model.pred)

	def DELETE(self,name):
		PredictorModel.delete(name)
		
	def POST(self, name):
		#In the public service this is a GET with a value argument, but it makes this internal service simpler to use a different HTTP method
		model = PredictorModel.load(name)
		if model is None: return web.notfound()
		request = json.loads(web.data())
		multipleValues = request.has_key('multiple') and request['multiple']
		instances = request['value'] if multipleValues else [ request['value'] ]
		if len(instances[0]) != model.featureCount:
			response = wrong_feature_count.copy()
			response.update( {'expected': model.featureCount, 'actual': len(instances[0]) } )
			return json.dumps(response)
		#FIXME Can we determine if predictor requires array (since most don't and it's additional processing overhead for no reason if that's the case)?
		values = model.predict( instances )
		response = { 'value': values if multipleValues else values[0] }
		return json.dumps(response)

class update:
	def POST(self, name):
		model = PredictorModel.load(name)
		if model is None: return web.notfound()
		updates = json.loads(web.data())['updates']
		sources = [ updates[i]['source'] for i in range(0,len(updates)) ]
		targets = [ updates[i]['target'] for i in range(0,len(updates)) ]
		model.pred.partial_fit(sources, model.labelsToInt(targets))
		model.replace(model.pred)
		return 'Predictor ' + name + ' updated OK'

if __name__ == '__main__':
	''' Set SKLEARN_PREDICTOR_DIR environment variable to path to directory in which to store predictors; otherwise will use current working directory. ''' 
	sys.stdout.write('Scikit-Learn micro service starting. Will store predictor models in ' +  PredictorModel.dir() + '\n')
	sys.stdout.flush()
	app = web.application(urls, globals())
	app.run()
