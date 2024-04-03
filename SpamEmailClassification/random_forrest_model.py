import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import prepared_data

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(prepared_data.X_train, prepared_data.y_train)

y_pred = rf_classifier.predict(prepared_data.X_test)

# Evaluate the accuracy of the classifier
accuracy = metrics.accuracy_score(prepared_data.y_test, y_pred)

print('Accuracy: ', accuracy)


model_filename = 'spam_email_rf_model.joblib'

if not os.path.exists(model_filename):
    dump(rf_classifier, model_filename)
else:
    user_input = input("The model file already exists. Do you want to overwrite it? (y/n): ")
    if user_input.lower() == 'y':
        dump(rf_classifier, model_filename)
        print("Model file overwritten.")
    else:
        print("Model file not overwritten.")
