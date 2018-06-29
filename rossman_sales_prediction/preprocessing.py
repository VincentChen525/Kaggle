import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
random.seed(42)

with open('train_data.pickle','rb') as f:
	train_data = pickle.load(f)
	num_records = len(train_data)
with open('store_data.pickle','rb') as f:
	store_data = pickle.load(f)

def feature_list(record):
	dt = datetime.strptime(record['Date'],'%Y-%m-%d')
	store_index = int(record['Store'])
	year = dt.year
	month = dt.month
	day = dt.day
	day_of_week = int(record['DayOfWeek'])
	try:
		store_open = int(record['Open'])
	except:
		store_open = 1
	promo = int(record['Promo'])

	return [store_open,store_index,day_of_week,promo,year,month,day]

train_data_X = []
train_data_y = []

for record in train_data:
	if record['Sales'] != '0':
		fl = feature_list(record)
		train_data_X.append(fl)
		train_data_y.append(int(record['Sales']))

print('train_data_X: ', train_data_X[:7])
full_X = train_data_X
full_X = np.array(full_X)
train_data_X = np.array(train_data_X)
print(train_data_X)
le = []
# 对每一列做labelencoding
for i in range(train_data_X.shape[1]):
	lec = LabelEncoder()
	lec.fit(full_X[:,i])
	# print('lec',lec)
	le.append(lec)
	train_data_X[:,i] = lec.transform(train_data_X[:,i])

# 将train 的labelencoder保存下来，方便test集的预处理
with open('le.pickle','wb') as f:
	pickle.dump(le,f,-1)

train_data_X = train_data_X.astype(int)
print('shape',train_data_X)
train_data_y = np.array(train_data_y)

with open('preprocessing_train_data.pickle','wb') as f:
	pickle.dump((train_data_X,train_data_y),f,-1)









