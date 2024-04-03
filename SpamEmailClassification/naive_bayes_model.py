import os
from sklearn import metrics, naive_bayes
from joblib import dump
import prepared_data

nb_classifier = naive_bayes.GaussianNB()

X_train = prepared_data.X_train.toarray()
X_test = prepared_data.X_test.toarray()

nb_classifier.fit(X_train, prepared_data.y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = metrics.accuracy_score(prepared_data.y_test, y_pred)

print('Accuracy: ', accuracy)

model_filename = 'spam_email_nb_model.joblib'

if not os.path.exists(model_filename):
    dump(nb_classifier, model_filename)
else:
    user_input = input("The model file already exists. Do you want to overwrite it? (y/n): ")
    if user_input.lower() == 'y':
        dump(nb_classifier, model_filename)
        print("Model file overwritten.")
    else:
        print("Model file not overwritten.")
