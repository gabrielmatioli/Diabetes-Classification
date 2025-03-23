import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

### Diabetes Classification ###

## Data preprocessing ##

pd.set_option('display.max_columns', 30)
df = pd.read_csv('../datasets/diabetes.csv')[:150000]
df.dropna(inplace=True)

ct = ColumnTransformer([('encoder', OrdinalEncoder(), [14, 19, 20, 21]),
                        ('scaler', StandardScaler(), [4, 15, 16])], remainder='passthrough')
transformed = ct.fit_transform(df)
extracted_columns = ct.get_feature_names_out()

df_transformed = pd.DataFrame(transformed, columns=extracted_columns)

# Class "1" has a 1468 datapoints, while "0" has 63170 and "2" has 10362
imbalanced_check = df_transformed['remainder__Diabetes_012'].value_counts()

# Removing class "1" due to limited influence on the dataset
df_transformed = df_transformed.loc[df_transformed['remainder__Diabetes_012'] != 1.0]

X, y = df_transformed.drop(columns=['remainder__Diabetes_012']), df_transformed['remainder__Diabetes_012']

# Oversampling the class "2" 
sampler = SMOTE()
X_sampled, y_sampled = sampler.fit_resample(X, y)

## Model training ##

X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.20, random_state=0)

clf = GradientBoostingClassifier(learning_rate=0.40, n_estimators=150, max_depth=4)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## Model testing & report ##

clf_report = classification_report(y_test, y_pred)
print('Classification Report: \n')
print(clf_report)

# Confusion Matrix Report
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap='magma')
plt.show()