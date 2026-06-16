from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,cardiac_arrest_prediction,detection_ratio,detection_accuracy

def login(request):

    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            enter = ClientRegister_Model.objects.get(username=username, password=password)
            request.session["userid"] = enter.id

            # ✅ GO TO PREDICTION PAGE AFTER LOGIN
            return redirect('Predict_Cardiac_Arrest_Type')

        except:
            return render(request, 'RUser/login.html', {'error': 'Invalid credentials'})

    return render(request, 'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Cardiac_Arrest_Type(request):

    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    if request.method == "POST":

        # ---------------------------
        # STEP 1: GET INPUT VALUES
        # ---------------------------
        Fid = request.POST.get('Fid')
        Age_In_Days = request.POST.get('Age_In_Days')
        Sex = request.POST.get('Sex')
        ChestPainType = request.POST.get('ChestPainType')
        RestingBP = request.POST.get('RestingBP')
        RestingECG = request.POST.get('RestingECG')
        MaxHR = request.POST.get('MaxHR')
        ExerciseAngina = request.POST.get('ExerciseAngina')
        Oldpeak = request.POST.get('Oldpeak')
        ST_Slope = request.POST.get('ST_Slope')
        slp = request.POST.get('slp')
        caa = request.POST.get('caa')
        thall = request.POST.get('thall')

        # ---------------------------
        # STEP 2: LOAD DATASET
        # ---------------------------
        data = pd.read_csv("Datasets.csv", encoding='latin-1')

        # Target creation
        data['Results'] = data['HeartDisease']

        # ---------------------------
        # STEP 3: SELECT FEATURES (IMPORTANT FIX)
        # ---------------------------
        features = [
            'Age_In_Days',
            'Sex',
            'ChestPainType',
            'RestingBP',
            'RestingECG',
            'MaxHR',
            'ExerciseAngina',
            'Oldpeak',
            'ST_Slope',
            'slp',
            'caa',
            'thall'
        ]

        x = data[features]
        y = data['Results']

        # ---------------------------
        # STEP 4: ENCODE TEXT DATA
        # ---------------------------
        le = LabelEncoder()

        for col in x.columns:
            if x[col].dtype == 'object':
                x[col] = le.fit_transform(x[col])

        # ---------------------------
        # STEP 5: SCALE DATA
        # ---------------------------
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        # ---------------------------
        # STEP 6: TRAIN TEST SPLIT
        # ---------------------------
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # ---------------------------
        # STEP 7: MODELS
        # ---------------------------
        svm_model = LinearSVC()
        rf_model = RandomForestClassifier()
        lr_model = LogisticRegression(max_iter=1000)
        mlp_model = MLPClassifier(max_iter=500)

        models = [
            ('svm', svm_model),
            ('rf', rf_model),
            ('lr', lr_model),
            ('mlp', mlp_model)
        ]

        classifier = VotingClassifier(estimators=models, voting='hard')

        classifier.fit(X_train, y_train)

        # ---------------------------
        # STEP 8: USER INPUT PREDICTION
        # ---------------------------
        user_data = [[
            Age_In_Days,
            Sex,
            ChestPainType,
            RestingBP,
            RestingECG,
            MaxHR,
            ExerciseAngina,
            Oldpeak,
            ST_Slope,
            slp,
            caa,
            thall
        ]]

        user_df = pd.DataFrame(user_data, columns=features)

        # encode user input same way
        for col in user_df.columns:
            if user_df[col].dtype == 'object':
                user_df[col] = le.fit_transform(user_df[col])

        user_df = scaler.transform(user_df)

        prediction = classifier.predict(user_df)

        # ---------------------------
        # STEP 9: RESULT
        # ---------------------------
        if prediction[0] == 0:
            val = "No Cardiac Arrest Found"
        else:
            val = "Cardiac Arrest Found"

        cardiac_arrest_prediction.objects.create(
            Fid=Fid,
            Age_In_Days=Age_In_Days,
            Sex=Sex,
            ChestPainType=ChestPainType,
            RestingBP=RestingBP,
            RestingECG=RestingECG,
            MaxHR=MaxHR,
            ExerciseAngina=ExerciseAngina,
            Oldpeak=Oldpeak,
            ST_Slope=ST_Slope,
            slp=slp,
            caa=caa,
            thall=thall,
            Prediction=val
        )

        return render(request, 'RUser/Predict_Cardiac_Arrest_Type.html', {'objs': val})

    return render(request, 'RUser/Predict_Cardiac_Arrest_Type.html')

