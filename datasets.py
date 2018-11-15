import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import collections

def load_csv(data_path):
    x = pd.read_excel(data_path)
    x.pop('SampleNO')
    y = x.pop('label')
    return x, y

def add_ab(X_no,y_no,X_ab,y_ab):
    rand_stat = random.randint(0,100)
    X_abm,_,y_abm,_ = train_test_split(X_ab,y_ab,test_size=0.2,random_state=rand_stat)
    X_r = X_no.append(X_abm, ignore_index=True)
    y_r = y_no.append(y_abm, ignore_index=True)

    X,y = shuffle(X_r,y_r,random_state=rand_stat)
    return X,y

def split_data(normal_path,abnormal_path):
    X_normal, y_normal = load_csv(normal_path)
    X_train, X_val, y_train, y_val = train_test_split(X_normal, y_normal, test_size=0.2, random_state=0)

    X_an, y_an = load_csv(abnormal_path)
    X_an_val,X_an_test,y_an_val,y_an_test = train_test_split(X_an, y_an,test_size=0.5,random_state=random.randint(0,200))

    X_val_8, X_test_8, y_val_8, y_test_8 = train_test_split(X_val, y_val, test_size=0.5, random_state=0)

    X_val_41, X_val_42, y_val_41, y_val_42 = train_test_split(X_val_8, y_val_8, test_size=0.5, random_state=0)
    X_val_21, X_val_22, y_val_21, y_val_22 = train_test_split(X_val_41, y_val_41, test_size=0.5, random_state=0)
    X_val_23, X_val_24, y_val_23, y_val_24 = train_test_split(X_val_42, y_val_42, test_size=0.5, random_state=0)
    
    X_test_41, X_test_42, y_test_41, y_test_42 = train_test_split(X_test_8, y_test_8, test_size=0.5, random_state=0)
    X_test_21, X_test_22, y_test_21, y_test_22 = train_test_split(X_test_41, y_test_41, test_size=0.5, random_state=0)
    X_test_23, X_test_24, y_test_23, y_test_24 = train_test_split(X_test_42, y_test_42, test_size=0.5, random_state=0)

    X_val_11, X_val_12, y_val_11, y_val_12 = train_test_split(X_val_21, y_val_21, test_size=0.5, random_state=0)
    X_val_13, X_val_14, y_val_13, y_val_14 = train_test_split(X_val_22, y_val_22, test_size=0.5, random_state=0)
    X_val_15, X_val_16, y_val_15, y_val_16 = train_test_split(X_val_23, y_val_23, test_size=0.5, random_state=0)
    X_val_17, X_val_18, y_val_17, y_val_18 = train_test_split(X_val_24, y_val_24, test_size=0.5, random_state=0)

    X_test_11, X_test_12, y_test_11, y_test_12 = train_test_split(X_test_21, y_test_21, test_size=0.5, random_state=0)
    X_test_13, X_test_14, y_test_13, y_test_14 = train_test_split(X_test_22, y_test_22, test_size=0.5, random_state=0)
    X_test_15, X_test_16, y_test_15, y_test_16 = train_test_split(X_test_23, y_test_23, test_size=0.5, random_state=0)
    X_test_17, X_test_18, y_test_17, y_test_18 = train_test_split(X_test_24, y_test_24, test_size=0.5, random_state=0)
    
    X_val1,y_val1 = add_ab(X_val_11,y_val_11,X_an_val, y_an_val)
    X_val2,y_val2 = add_ab(X_val_12,y_val_12,X_an_val, y_an_val)
    X_val3,y_val3 = add_ab(X_val_13,y_val_13,X_an_val, y_an_val)
    X_val4,y_val4 = add_ab(X_val_14,y_val_14,X_an_val, y_an_val)
    X_val5,y_val5 = add_ab(X_val_15,y_val_15,X_an_val, y_an_val)
    X_val6,y_val6 = add_ab(X_val_16,y_val_16,X_an_val, y_an_val)
    X_val7,y_val7 = add_ab(X_val_17,y_val_17,X_an_val, y_an_val)
    X_val8,y_val8 = add_ab(X_val_18,y_val_18,X_an_val, y_an_val)
    
    X_test1,y_test1 = add_ab(X_test_11,y_test_11,X_an_test, y_an_test)
    X_test2,y_test2 = add_ab(X_test_12,y_test_12,X_an_test, y_an_test)
    X_test3,y_test3 = add_ab(X_test_13,y_test_13,X_an_test, y_an_test)
    X_test4,y_test4 = add_ab(X_test_14,y_test_14,X_an_test, y_an_test)
    X_test5,y_test5 = add_ab(X_test_15,y_test_15,X_an_test, y_an_test)
    X_test6,y_test6 = add_ab(X_test_16,y_test_16,X_an_test, y_an_test)
    X_test7,y_test7 = add_ab(X_test_17,y_test_17,X_an_test, y_an_test)
    X_test8,y_test8 = add_ab(X_test_18,y_test_18,X_an_test, y_an_test)
    
    data = collections.namedtuple('data', 'X y')    
    val_set = [data(X_val1,y_val1),data(X_val2,y_val2),data(X_val3,y_val3),data(X_val4,y_val4),data(X_val5,y_val5),data(X_val6,y_val6),data(X_val7,y_val7),data(X_val8,y_val8)]
    test_set = [data(X_test1,y_test1),data(X_test2,y_test2),data(X_test3,y_test3),data(X_test4,y_test4),data(X_test5,y_test5),data(X_test6,y_test6),data(X_test7,y_test7),data(X_test8,y_test8)]

    return X_train, X_val,val_set,test_set


def isolation_data(normal_path,abnormal_path):
    X_normal, y_normal = load_csv(normal_path)
    X_train_no, X_val_no, y_train_no, y_val_no = train_test_split(X_normal, y_normal, test_size=0.3, random_state=0)
    X_val_no, X_test_no, y_val_no, y_test_no = train_test_split(X_val_no, y_val_no, test_size=0.5, random_state=0)
    X_abnormal, y_abnormal = load_csv(abnormal_path)
    X_train_ab, X_val_ab, y_train_ab, y_val_ab = train_test_split(X_abnormal, y_abnormal, test_size=0.3, random_state=0)
    X_val_ab, X_test_ab, y_val_ab, y_test_ab = train_test_split(X_val_ab, y_val_ab, test_size=0.5, random_state=0)
    X_train = X_train_no.append(X_train_ab, ignore_index=True)
    y_train = y_train_no.append(y_train_ab, ignore_index=True)
    X_val = X_val_no.append(X_val_ab, ignore_index=True)
    y_val = y_val_no.append(y_val_ab, ignore_index=True)
    X_test = X_test_no.append(X_test_ab, ignore_index=True)
    y_test = y_test_no.append(y_test_ab, ignore_index=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_data(model_type,normal_path,abnormal_path):
    if model_type == 'OneClassSVM':
        return split_data(normal_path, abnormal_path)
    if model_type == 'isolation':
        return isolation_data(normal_path, abnormal_path)


				
