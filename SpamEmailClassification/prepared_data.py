import pandas as pd
import email
import os
from bs4 import BeautifulSoup
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class EmailDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        columns = ['Sender', 'Recipient', 'Subject', 'Date', 'Content']
        super().__init__(*args, columns=columns, **kwargs)


def extract_plain_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    plain_text = soup.get_text(separator=' ')
    plain_text = plain_text.replace('\n', '')
    return plain_text


def extract_emails_from_folder(folder_path):
    emails_df = EmailDataFrame()

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'rb') as file:
            msg = email.message_from_binary_file(file)

        # Extract email content, subject, or other relevant information
        sender = msg['From']
        recipient = msg['To']
        subject = msg['Subject']
        date = msg['Date']

        # Extract email content
        email_content = ''
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    email_content += part.get_payload()
                elif part.get_content_type() == 'text/html':
                    email_content = extract_plain_text(part.get_payload())

                    # html_content = part.get_payload()
                    # soup = BeautifulSoup(html_content, 'html.parser')
                    # email_content += soup.get_text(separator=' ')
        else:
            email_content = extract_plain_text(msg.get_payload())

        # Append email information to the DataFrame
        email_df = pd.DataFrame({
            'Sender': [sender],
            'Recipient': [recipient],
            'Subject': [subject],
            'Date': [date],
            'Content': [email_content]
        })

        # Concatenate the temporary DataFrame with the main DataFrame
        emails_df = pd.concat([emails_df, email_df], ignore_index=True)

    return emails_df


easy_ham_1_emails = extract_emails_from_folder('easy_ham_1')
easy_ham_2_emails = extract_emails_from_folder('easy_ham_2')
hard_ham_1_emails = extract_emails_from_folder('hard_ham_1')
spam_1_emails = extract_emails_from_folder('spam_1')
spam_2_emails = extract_emails_from_folder('spam_2')

real_emails = pd.concat([easy_ham_1_emails, easy_ham_2_emails, hard_ham_1_emails], ignore_index=True)
spam_emails = pd.concat([spam_1_emails, spam_2_emails], ignore_index=True)

spam_emails['Label'] = 'spam'
real_emails['Label'] = 'real'

all_emails = pd.concat([real_emails, spam_emails], ignore_index=True)
all_emails = all_emails.sample(frac=1).reset_index(drop=True)

X = all_emails[['Content', 'Subject', 'Date', 'Recipient', 'Sender']]
y = all_emails['Label']

X.loc[:, 'Content'] = X['Content'].astype(str)
X.loc[:, 'Subject'] = X['Subject'].astype(str)
X.loc[:, 'Date'] = X['Date'].astype(str)
X.loc[:, 'Recipient'] = X['Recipient'].astype(str)
X.loc[:, 'Sender'] = X['Sender'].astype(str)

vectorizers = [('vectorizer_'+str(i), CountVectorizer(), i) for i in range(X.shape[1])]
vectorizer = ColumnTransformer(vectorizers)
X_tokenized = vectorizer.fit_transform(X.values)

dump(vectorizer, 'vectorizer.joblib')

X_train, X_test, y_train, y_test = train_test_split(X_tokenized, y, test_size=0.2, random_state=42)
