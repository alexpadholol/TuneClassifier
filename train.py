from hyperopt import fmin, tpe, hp,STATUS_OK
from model import *
import datasets
from sklearn.preprocessing import StandardScaler
import pickle
import os

model_name = 'OneClassSVM'
#nu = 0.0005
kernel = 'rbf'
#gamma = 0.0002
#training path
model_home = os.getcwd()
normal_path = model_home+'/datasets/normal2.xlsx'
abnormal_path = model_home+'/datasets/abnormal.xlsx'

print(abnormal_path)

X_train,X_val,val_set,test_set = datasets.get_data(model_name,normal_path,abnormal_path)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

def fn(args):
	nu,gamma = args
	params = {'nu':nu,'kernel':kernel,'gamma':gamma}
	oneclasssvm = creat_model(model_name,params)
	oneclasssvm.fit(X_train_std)

	errors = 0
	for data in val_set:
		X = data.X
		y = data.y
		#print(y)
		X_std = sc.transform(X)
		y_pred = oneclasssvm.predict(X_std)
		scores = oneclasssvm.score_samples(X_std)
		eval_r = evaluate(y,y_pred,scores)
		auc = eval_r['AUC']
		Specificity = eval_r['Specificity']
		Sensitivity = eval_r['Sensitivity']
		errors = errors + 1 - (3 * auc + 2 * Specificity + 5 * Sensitivity) / 10
	#print(auc)
	return errors

best = fmin(fn,space=[hp.uniform('nu',0.0001 , 0.01),hp.uniform('gamma',0.0001 , 0.05)],algo=tpe.suggest,max_evals=120000)

nu = best['nu']
gamma = best['gamma']
kernel = 'rbf'

param = {'nu':nu,'kernel':kernel,'gamma':gamma}

clf = creat_model(model_name,param)
clf.fit(X_train_std)
#y_val_pred = clf.predict(X_val_std)
#scores = clf.score_samples( X_val_std)
#eval_r = evaluate(y_val,y_val_pred,scores)

f = open(model_home+"/model/out.txt", "a")

f.write('val result:\n')

for data in val_set:
	X = data.X
	y = data.y
	X_std = sc.transform(X)
	y_pred = clf.predict(X_std)
	scores = clf.score_samples(X_std)
	eval_r = evaluate(y,y_pred,scores)
	for key in eval_r.keys():
		f.write(key+':')
		f.write(str(eval_r[key]))
		f.write('\n')
	f.write('\n')			
#f.write('auc:')

f.write('test result:\n')

for data in test_set:
        X = data.X
        y = data.y
        X_std = sc.transform(X)
        y_pred = clf.predict(X_std)
        scores = clf.score_samples(X_std)
        eval_r = evaluate(y,y_pred,scores)
        for key in eval_r.keys():
                f.write(key+':')
                f.write(str(eval_r[key]))
                f.write('\n')
        f.write('\n')

f.write('\n')

for key in best.keys():
	f.write(key+':')
	f.write(str(best[key]))
	f.write('\n')
f.write('\n')
f.close() 

model_home = model_home+'/model/'
#print(abnormal_path)

sc_path = model_home + 'sc.pk'
model_path = model_home + 'model.pk'

sc_file = open(sc_path,'wb')
model_file = open(model_path,'wb')

nu,gamma = best['nu'],best['gamma']
param = {'nu':nu,'kernel':kernel,'gamma':gamma}

clf = creat_model(model_name,param)
clf.fit(X_train_std)

pickle.dump(sc,sc_file)
sc_file.close()

pickle.dump(clf,model_file)
model_file.close()

print('done!')
