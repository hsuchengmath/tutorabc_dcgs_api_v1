

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics 
from sklearn.metrics import confusion_matrix
import random
import matplotlib.pyplot as plt  
import numpy as np
from sklearn import metrics


def transform_date_to_age(date_str, categorical=True):
    if date_str != 'None':
        age_val = 2021 - pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S').year
        if categorical is False:
            return age_val
        else:
            if age_val <= 30:
                return '0~30'
            elif age_val > 30 and age_val < 50:
                return '30~50'
            else:
                return '50~'   
    else:
        return 'None'


def train_model(train_data,label):
    train_data = np.array(train_data)
    rf = RandomForestRegressor()
    rf.fit(train_data, label)
    return rf


def predict_score(pred_prob, Y_test_array, binary_threshold=0.5):
    pred_one_hot = list()
    for i in range(pred_prob.shape[0]):
        if pred_prob[i] >= binary_threshold:
            pred_one_hot.append(1)
        else:
            pred_one_hot.append(0)
    print(metrics.classification_report(list(Y_test_array), pred_one_hot))
    print('---------------------------------------')
    print('Confusion Matrix')
    print(np.transpose(confusion_matrix(list(Y_test_array), pred_one_hot).T))
    print('---------------------------------------')
    print('positive label : 1 | negative label : 0')


def random_model_performance(ground_truth):
    ground_truth = list(ground_truth)
    pos,neg = 0,0
    for val in ground_truth:
        if int(val) == 1:
            pos +=1
        else:
            neg +=1
    random_pred = np.array([random.sample([1,0],1)[0] for _ in range(len(ground_truth))])
    predict_score(random_pred,ground_truth)


def ROC_curve_plot(y_with_pred_list, model_name_list, model_color_list):
    '''
    import matplotlib.pyplot as plt  
    import numpy as np
    from sklearn import metrics
    y = np.array([0, 0, 1, 1,0,0])
    pred = np.array([0.1, 0.4, 0.35, 0.8,0.2,0.1])
    pred2 = np.array([0.9, 0.6, 0.35, 0.8,0.2,0.9])
    y_with_pred_list = [(y, pred),(y,pred2)]
    model_name_list = ['NCF','random']
    model_color_list = ['crimson','lightskyblue']
    ROC_curve_plot(y_with_pred_list, model_name_list, model_color_list)
    '''
    import matplotlib.pyplot as plt  
    import numpy as np
    from sklearn import metrics
    plt.figure()
    for i in range(len(model_name_list)):
        model_name, y_with_pred, used_color = model_name_list[i], y_with_pred_list[i], model_color_list[i]
        y, pred = y_with_pred[0], y_with_pred[1]
        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color=used_color, label=model_name+'-ROC curve (area = %0.2f) '% roc_auc)  
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")
    plt.show()


def DET_curve_plot(y_with_pred_list, model_name_list, model_color_list):
    '''
    import matplotlib.pyplot as plt  
    import numpy as np
    from sklearn import metrics
    y = np.array([0, 0, 1, 1,0,0])
    pred = np.array([0.1, 0.4, 0.35, 0.8,0.2,0.1])
    pred2 = np.array([0.9, 0.6, 0.35, 0.8,0.2,0.9])
    y_with_pred_list = [(y, pred),(y,pred2)]
    model_name_list = ['NCF','random']
    model_color_list = ['crimson','lightskyblue']
    DET_curve_plot(y_with_pred_list, model_name_list, model_color_list)
    '''
    import matplotlib.pyplot as plt  
    import numpy as np
    from sklearn import metrics
    plt.figure()
    for i in range(len(model_name_list)):
        model_name, y_with_pred, used_color = model_name_list[i], y_with_pred_list[i], model_color_list[i]
        y, pred = y_with_pred[0], y_with_pred[1]
        fpr, fnr, thresholds = metrics.det_curve(y, pred)
        plt.plot(fpr, fnr, color=used_color, label=model_name)  
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Detection Error Tradeoff curve')
    plt.legend(loc="lower right")
    plt.show()