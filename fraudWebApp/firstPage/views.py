from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, LabelEncoder

# Create your views here.

import joblib
reloadModel=joblib.load('./models/RFModelforfraud_2.pkl')

def index(request):
    context={'a':'Hello'}
    return render(request, 'index.html', context)
    #return HttpResponse({'a':1})

# def df_edit_2(fraud_dataframe):
#         label = LabelEncoder()
#         fraud_dataframe['nameOrig'] = label.fit_transform(fraud_dataframe['nameOrig']) #encoding the nameorig
#         fraud_dataframe['nameDest']= label.fit_transform(fraud_dataframe['nameDest'])
#         fraud_dataframe['type']= label.fit_transform(fraud_dataframe['type'])#encoding the namedest
#         df1 = pd.get_dummies(fraud_dataframe) # making dummies on the type of transaction
#         #features = df1.drop(['isFraud','isFlaggedFraud'],axis=1)
#         #preprocessing the data using scalers
#         scaler = MinMaxScaler() # to reduce higher spread of dimentionality and outliers
#         scaler.fit(df1)
#         featuress = scaler.transform(df1)
#         feat_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(featuress) #adding polynomial features
#         return feat_poly 

def fraudDetection(request):
    print(request)
    # if request.method == 'GET':
    #     temp={}
    #     temp['step']=100 #request.GET.get('stepVal')
    #     temp['type']= 'TRANSFER' #request.GET.get('typeVal')
    #     temp['amount']= 47664 #request.GET.get('amountVal')
    #     temp['nameOrig']= 'C1720120297' #request.GET.get('nameOrigVal')
    #     temp['oldbalanceOrg']= 500000 #request.GET.get('oldbalanceOrgVal')
    #     temp['newbalanceOrig']= 100664 #request.GET.get('newbalanceOrigVal')
    #     temp['nameDest']= 'C496863720' #request.GET.get('nameDestVal')
    #     temp['oldbalanceDest']= 1605021.18 #request.GET.get('oldbalanceDestVal')
    #     temp['newbalanceDest']=1000000.89 #request.GET.get('newbalanceDestVal')
    if request.method == 'GET':
        temp={}
        temp['step']=request.GET.get('stepVal')
        temp['type']=request.GET.get('typeVal')
        temp['amount']= request.GET.get('amountVal')
        temp['nameOrig']= request.GET.get('nameOrigVal')
        temp['oldbalanceOrg']= request.GET.get('oldbalanceOrgVal')
        temp['newbalanceOrig']= request.GET.get('newbalanceOrigVal')
        temp['nameDest']= request.GET.get('nameDestVal')
        temp['oldbalanceDest']= request.GET.get('oldbalanceDestVal')
        temp['newbalanceDest']= request.GET.get('newbalanceDestVal')
    testDtaa=pd.DataFrame({'x':temp}).transpose()
    label = LabelEncoder()
    testDtaa['nameOrig'] = label.fit_transform(testDtaa['nameOrig']) #encoding the nameorig
    testDtaa['nameDest']= label.fit_transform(testDtaa['nameDest'])
    testDtaa['type']= label.fit_transform(testDtaa['type'])#encoding the namedest
    scaler = MinMaxScaler() # to reduce higher spread of dimentionality and outliers
    scaler.fit(testDtaa)
    featuress = scaler.transform(testDtaa)
    feat_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(featuress) #adding polynomial features
    scoreval = reloadModel.predict(feat_poly)[0]
    context={'scoreval':scoreval}
    return render(request, 'index.html', context)