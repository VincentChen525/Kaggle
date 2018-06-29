import pickle
import numpy
numpy.random.seed(42)
from model import *
from sklearn.preprocessing import OneHotEncoder
import sys
sys.setrecursionlimit(100)

# initialize parameter
train_ratio = 0.9
shuffle_data = False
one_hot_as_input = False
embeddings_as_input = False
save_embeddings = True
saved_embeddings_fname = 'embeddings.pickle'

f = open('preprocessing_train_data.pickle','rb')
(X,y) = pickle.load(f)

num_records = len(X)
train_size = int(train_ratio * num_records)

if shuffle_data:
	print("Using shuffle data")
	sh = numpy.arange(X.shape[0])
	numpy.random.shuffle(sh)
	X = X[sh]
	y = y[sh]

if embeddings_as_input:
	print("Using learned embeddings as input")
	X = embed_features(X, saved_embeddings_fname)

if one_hot_as_input:
	print("Using one-hot encoding as input")
	enc = OneHotEncoder(sparse = False)
	enc.fit(X)
	X = enc.transform(X)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:]
y_val = y[train_size:]

def sample(X,y,n):
	# random samples
	num_row = X.shape[0]
	indices = numpy.random.randint(num_row, size = n)
	return X[indices,:],y[indices]

# sample data sparsity
X_train,y_train = sample(X_train,y_train,200) 
print("sample data: ",X_train)

models = []

print('Fitting NN_with_Entity_Embedding....')
for i in range(5):
	models.append(NN_with_EntityEmbedding(X_train,y_train,X_val,y_val))

if save_embeddings:
	model = models[0].model
	store_embedding = model.get_layer('store_embedding').get_weights()[0]
	dow_embedding = model.get_layer('dow_embedding').get_weights()[0]
	year_embedding = model.get_layer('year_embedding').get_weights()[0]
	month_embedding = model.get_layer('month_embedding').get_weights()[0]
	day_embedding = model.get_layer('day_embedding').get_weights()[0]
	with open(saved_embeddings_fname,'wb') as f:
		pickle.dump([store_embedding,dow_embedding,year_embedding,month_embedding,day_embedding],f,-1)

def evaluate_models(models,X,y):
	assert(min(y) > 0)
	pred_sales = numpy.array([model.guess(X) for model in models])
	mean_sales = pred_sales.mean(axis = 0)
	relative_err = numpy.absolute((y - mean_sales) / y)
	result = numpy.sum(relative_err) / len(y)
	return result

print('Evaluate combined models....')
print('Training Error...')
err_train = evaluate_models(models,X_train,y_train)
print(err_train)

print('Validation Error...')
err_val = evaluate_models(models,X_val,y_val)
print(err_val)







