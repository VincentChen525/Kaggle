import pickle
import csv

def csv2dict(file):
	data = []
	features = []
	for index,row in enumerate(file):
		if index == 0:
			features = row
			continue
		data.append({feature:value for feature, value in zip(features,row)})
	return data

def fillNan(data, replace = '0'):
	for index, x in enumerate(data):
		for feature, val in x.items():
			if val == '':
				x[feature] = replace
		data[index] = x

train_data = 'train1.csv'
store_data = 'store.csv'

with open(train_data,encoding='utf-8-sig') as f:
	data = csv.reader(f, delimiter = ',')
	with open('train_data.pickle','wb') as ff:
		data = csv2dict(data)
		data = data[::-1]
		pickle.dump(data,ff,-1)
		print(data[:3])

with open(store_data) as f:
	data = csv.reader(f, delimiter = ',')
	with open('store_data.pickle','wb') as ff:
		data = csv2dict(data)
		fillNan(data)
		pickle.dump(data,ff,-1)
		print(data[:2])


