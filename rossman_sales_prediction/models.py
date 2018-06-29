import numpy
import pickle
numpy.random.seed(123)

from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint

# 提取在NN的embedding层处理的数据作为embedding 特征
def embed_features(X, saved_embeddings_fname):
	f_embeddings = open(saved_embeddings_fname,'rb')
	embeddings = pickle.load(f_embeddings)

	index_embedding_mapping = {1:0,2:1,4:2,5:3,6:4}
	X_embedded = []
	(num_samples,num_features) = X.shape
	for sample in X:
		embedded_features = []
		for feature_index,val in enumerate(sample):
			if feature_index not in index_embedding_mapping:
				embedded_features += [int(val)]
			else:
				emb_index = index_embedding_mapping[feature_index]
				embedded_features += embeddings[emb_index][int(val)].tolist()
		X_embedded.append(embedded_features)
	return numpy.array(X_embedded)

def split_features(X):
    X_list = []

    store_index = X[..., [1]]
    X_list.append(store_index)

    day_of_week = X[..., [2]]
    X_list.append(day_of_week)

    promo = X[..., [3]]
    X_list.append(promo)

    year = X[..., [4]]
    X_list.append(year)

    month = X[..., [5]]
    X_list.append(month)

    day = X[..., [6]]
    X_list.append(day)

    return X_list

class Model(object):
	def evaluate(self,X_val,y_val):
		assert(min(y_val) > 0)
        pred_sales = self.prediction(X_val)
        relative_err = numpy.absolute((y_val - pred_sales) / y_val)
        result = numpy.sum(relative_err) / len(y_val)
        return result

class NN_with_EntityEmbedding(Model):
	def __init__(self,X_train,y_train,X_val,y_val):
		super().__init__()
		self.epochs = 10
		self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", 
			verbose=1, save_best_only=True)
		self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
		self.__build_keras_model()
		self.fit(X_train,y_train,X_val,y_val)

	def preprocessing(self,X):
		X_list = split_features(X)
		return X_list

	def __build_keras_model(self):
		# 嵌入层能针对不同的列做不同的处理
		input_store = Input(shape=(1,))
		output_store = Embedding(1115, 10, name = 'store_embedding')(input_store)
		output_store = Reshape(target_shape = (10,))(output_store)

		input_dow = Input(shape=(1,))
		output_dow = Embedding(7, 6, name = 'dow_embedding')(input_dow)
		output_dow = Reshape(target_shape = (6,))(output_dow)

		# promo的值只有1或者0
		input_promo = Input(shape=(1,))
		output_promo = Dense(1)(input_promo)

		input_year = Input(shape=(1,))
		output_year = Embedding(3, 2, name = 'year_embedding')(input_year)
		output_year = Reshape(target_shape = (2,))(output_year)

		input_month = Input(shape=(1,))
		output_month = Embedding(12, 6, name = 'month_embedding')(input_month)
		output_month = Reshape(target_shape = (6,))(output_month)

		input_day = Input(shape=(1,))
		output_day = Embedding(31, 10, name = 'day_embedding')(input_day)
		output_day = Reshape(target_shape = (10,))(output_day)

		input_model = [input_store,input_dow,input_promo,input_year,input_month,input_day]

		output_embeddings = [output_store,output_dow,output_promo,output_year,output_month,output_day]

		output_model = Concatenate()(output_embeddings)
		output_model = Dense(1000, kernel_initialize = 'uniform')(output_model)
		output_model = Activation('relu')(output_model)
		output_model = Dense(500, kernel_initialize = 'uniform')(output_model)
		output_model = Activation('relu')(output_model)
		output_model = Dense(1)(output_model)
		output_model = Activation('sigmoid')(output_model)

		self.model = KerasModel(inputs = input_model,outputs = output_model)
		self.model.compile(loss = 'mean_absolute_error', optimizer = 'adam')

	def _val_for_fit(self, val):
		val = numpy.log(val) / self.max_log_y
		return val

	def _val_for_pred(self, val):
		return numpy.exp(val * self.max_log_y)

	def fit(self, X_train,y_train,X_val,y_val):
		self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train)),
		validation_data = (self.preprocessing(X_val), self._val_for_fit(y_val)),
		epochs = self.epochs, batch_size = 128)
		print('Result on validation data: ', self.evaluate(X_val,y_val))

	def prediction(self, X):
		result = self.model.predict(self.preprocessing(X)).flatten()
		return self._val_for_pred(result)

		




