import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
targets = train.Survived
train.drop('Survived',axis=1,inplace=True)
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop('index', inplace=True,axis=1)

# we extract the title from each name
combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
# a map of more aggregated titles
Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }

# we map each title
combined['Title'] = combined.Title.map(Title_Dictionary)

grouped_train = combined.head(891).groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()

grouped_test = combined.iloc[891:].groupby(['Sex','Pclass','Title'])
grouped_median_test = grouped_test.median()

grouped_median_train

def fillAges(row, grouped_median):
        if row['Sex']=='female' and row['Pclass']==1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female',1,'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female',1,'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female',1,'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female',1,'Royalty']['Age']
        elif row['Sex']=='female' and row['Pclass']==2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female',2,'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female',2,'Mrs']['Age']
        elif row['Sex']=='female' and row['Pclass']==3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female',3,'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female',3,'Mrs']['Age']
        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']
        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Officer']['Age']
        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']

combined.head(891).Age = combined.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age'])
                                                      else r['Age'], axis=1)

combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age'])
                                                      else r['Age'], axis=1)


#combined.info()

combined.head()

combined.drop(['PassengerId','Name', 'Cabin','Embarked','SibSp','Parch','Ticket'],axis=1,inplace=True)

combined.head()

sex_dummies = pd.get_dummies(combined['Sex'],prefix='Sex')
combined = pd.concat([combined,sex_dummies], axis=1)
combined.drop('Sex',axis=1, inplace=True)

pclass_dummies = pd.get_dummies(combined['Pclass'],prefix='Pclass')
combined = pd.concat([combined, pclass_dummies],axis=1)
combined.drop('Pclass',axis=1,inplace=True)

#combined.drop('Title',axis=1,inplace=True)
title_dummies = pd.get_dummies(combined['Title'],prefix='Title')
combined = pd.concat([combined, title_dummies],axis=1)
combined.drop('Title',axis=1,inplace=True)

print(combined.head(),'\n\n',combined.info())

combined = (combined-combined.mean())/combined.std()
