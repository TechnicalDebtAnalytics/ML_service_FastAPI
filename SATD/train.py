import os
import re
import ast
import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


DATASET_PATH = "dataset/satd_dataset.csv"
KEYWORD_DIR = "dataset"
MODEL_DIR = "model"



# =====================================================
# Load SATD External Features
# =====================================================


def load_feature_weights():


    files = {

        "code/design_debt":
            "code_and_design_debt.txt",

        "documentation_debt":
            "documentation_debt.txt",

        "requirement_debt":
            "requirements_debt.txt",

        "test_debt":
            "test_debt.txt"

    }



    feature_weights = {}



    for category, filename in files.items():


        feature_weights[category] = {}


        path = os.path.join(
            KEYWORD_DIR,
            filename
        )


        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:


            for line in file:


                try:

                    score, feature = line.split("->")


                    score=float(
                        score.strip()
                    )


                    tokens = ast.literal_eval(
                        feature.strip()
                    )


                    phrase=" ".join(tokens).lower()


                    feature_weights[category][phrase]=score


                except:

                    continue



    return feature_weights




satd_features = load_feature_weights()



print("\nLoaded SATD Features")


for k,v in satd_features.items():

    print(
        k,
        len(v)
    )





# =====================================================
# Dataset
# =====================================================


df=pd.read_csv(
    DATASET_PATH
)



df=df.rename(
    columns={
        "text":"comment"
    }
)



df=df[
    [
        "comment",
        "classification"
    ]
].dropna()



df=df.drop_duplicates(
    subset=["comment"]
)



print("\nOriginal Distribution")

print(
    df.classification.value_counts()
)





# =====================================================
# Reduce Non Debt
# =====================================================


non_debt=df[
    df.classification=="non_debt"
]


debt=df[
    df.classification!="non_debt"
]



if len(non_debt)>12000:

    non_debt=non_debt.sample(
        12000,
        random_state=42
    )



df=pd.concat(
    [
        non_debt,
        debt
    ],
    ignore_index=True
)



print("\nAfter Reduction")

print(
    df.classification.value_counts()
)






# =====================================================
# Cleaning
# =====================================================


def clean_comment(text):


    text=str(text).lower()


    text=re.sub(
        r"http\S+",
        " ",
        text
    )


    text=re.sub(
        r"//|/\*|\*/|\*",
        " ",
        text
    )


    text=re.sub(
        r"[^a-z0-9_\s]",
        " ",
        text
    )


    text=re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()




df["clean_comment"]=df.comment.apply(
    clean_comment
)







# =====================================================
# SATD Feature Extraction
# =====================================================


def extract_satd_features(text):


    features=[]



    for category,keywords in satd_features.items():


        score=0



        for phrase,weight in keywords.items():

            if phrase in text:

                # increase importance
                score += weight * 5



        features.append(score)



    # Binary indicators

    for category,keywords in satd_features.items():


        found=0


        for phrase in keywords:


            if phrase in text:

                found=1
                break



        features.append(found)



    return features






satd_matrix = csr_matrix(

    df.clean_comment.apply(
        extract_satd_features
    ).tolist()

)





print(
    "\nSATD feature size:",
    satd_matrix.shape
)






# =====================================================
# Labels
# =====================================================


X=df.clean_comment



encoder=LabelEncoder()



y=encoder.fit_transform(
    df.classification
)



print("\nClass Mapping")


for i,label in enumerate(
    encoder.classes_
):

    print(
        i,
        "=",
        label
    )






# =====================================================
# Split
# =====================================================


(
X_train,
X_test,
y_train,
y_test,
satd_train,
satd_test

)=train_test_split(

    X,
    y,
    satd_matrix,

    test_size=0.2,

    random_state=42,

    stratify=y

)







# =====================================================
# Word TF-IDF
# =====================================================


word_vectorizer=TfidfVectorizer(

    analyzer="word",

    ngram_range=(1,3),

    max_features=120000,

    min_df=2,

    sublinear_tf=True

)



X_train_word=word_vectorizer.fit_transform(
    X_train
)



X_test_word=word_vectorizer.transform(
    X_test
)






# =====================================================
# Character TF-IDF
# =====================================================


char_vectorizer=TfidfVectorizer(

    analyzer="char",

    ngram_range=(2,6),

    max_features=150000,

    min_df=2,

    sublinear_tf=True

)



X_train_char=char_vectorizer.fit_transform(
    X_train
)



X_test_char=char_vectorizer.transform(
    X_test
)






# =====================================================
# Combine
# =====================================================


X_train_final=hstack(

    [
        X_train_word,
        X_train_char,
        satd_train
    ]

)



X_test_final=hstack(

    [
        X_test_word,
        X_test_char,
        satd_test
    ]

)



print(
    "\nFinal Features:",
    X_train_final.shape
)






# =====================================================
# Linear SVM
# =====================================================


model=LinearSVC(

    C=2.0,


    class_weight={

        0:1.5,   # code/design

        1:5.0,   # defect

        2:5.0,   # documentation

        3:0.4,   # non debt

        4:5.0,   # requirement

        5:5.0    # test

    },


    max_iter=50000,

    random_state=42

)




model.fit(

    X_train_final,

    y_train

)






# =====================================================
# Evaluation
# =====================================================


pred=model.predict(
    X_test_final
)



print("\nAccuracy")

print(
    accuracy_score(
        y_test,
        pred
    )
)



print("\nMacro F1")

print(
    f1_score(
        y_test,
        pred,
        average="macro"
    )
)



print("\nWeighted F1")

print(
    f1_score(
        y_test,
        pred,
        average="weighted"
    )
)



print("\nReport")


print(

classification_report(

    y_test,

    pred,

    target_names=encoder.classes_,

    digits=4,

    zero_division=0

)

)



print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        pred
    )
)






# =====================================================
# Save
# =====================================================


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



joblib.dump(
    model,
    MODEL_DIR+"/svm_satd_model.pkl"
)



joblib.dump(
    word_vectorizer,
    MODEL_DIR+"/word_tfidf.pkl"
)



joblib.dump(
    char_vectorizer,
    MODEL_DIR+"/char_tfidf.pkl"
)



joblib.dump(
    encoder,
    MODEL_DIR+"/label_encoder.pkl"
)



joblib.dump(
    satd_features,
    MODEL_DIR+"/satd_features.pkl"
)



print("\nTraining completed successfully")


"""
Loaded SATD Features
code/design_debt 501
documentation_debt 509
requirement_debt 509
test_debt 504

Original Distribution
classification
non_debt              35522
code/design_debt       2262
requirement_debt        550
defect_debt             351
test_debt                81
documentation_debt       49
Name: count, dtype: int64

After Reduction
classification
non_debt              12000
code/design_debt       2262
requirement_debt        550
defect_debt             351
test_debt                81
documentation_debt       49
Name: count, dtype: int64

SATD feature size: (15293, 8)

Class Mapping
0 = code/design_debt
1 = defect_debt
2 = documentation_debt
3 = non_debt
4 = requirement_debt
5 = test_debt

Final Features: (12234, 164990)

Accuracy
0.8950637463223275

Macro F1
0.592368601349882

Weighted F1
0.8877313352843322

Report
                    precision    recall  f1-score   support

  code/design_debt     0.7020    0.7594    0.7296       453
       defect_debt     0.4054    0.2143    0.2804        70
documentation_debt     1.0000    0.5000    0.6667        10
          non_debt     0.9535    0.9733    0.9633      2400
  requirement_debt     0.4615    0.2727    0.3429       110
         test_debt     0.6667    0.5000    0.5714        16

          accuracy                         0.8951      3059
         macro avg     0.6982    0.5366    0.5924      3059
      weighted avg     0.8847    0.8951    0.8877      3059


Confusion Matrix
[[ 344   10    0   76   22    1]
 [  31   15    0   20    2    2]
 [   4    0    5    1    0    0]
 [  52    5    0 2336    6    1]
 [  59    6    0   15   30    0]
 [   0    1    0    2    5    8]]

Training completed successfully"""