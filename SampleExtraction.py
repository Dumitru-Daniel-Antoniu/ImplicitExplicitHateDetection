import pandas as pd

df = pd.read_csv("C:\\Users\\ddumi\\Desktop\\Faculty\\MasterFirstYear\\SecondSemester\\StatisticalProcessingOfNaturalLanguage\\ProjectResources\\DynaHate.csv")

hate_train = df[(df['label'] == 'hate') & (df['split'] == 'train')]
hate_test = df[(df['label'] == 'hate') & (df['split'] == 'test')]
nothate_train = df[(df['label'] == 'nothate') & (df['split'] == 'train')]
nothate_test = df[(df['label'] == 'nothate') & (df['split'] == 'test')]

print(len(hate_train))
print(len(nothate_train))
print(len(hate_test))
print(len(nothate_test))

hate_train_sample = hate_train.sample(n=8000, random_state=42)
hate_test_sample = hate_test.sample(n=2268, random_state=42)
nothate_train_sample = nothate_train.sample(n=8000, random_state=42)
nothate_test_sample = nothate_test.sample(n=1852, random_state=42)

final_df = pd.concat([
    hate_train_sample,
    hate_test_sample,
    nothate_train_sample,
    nothate_test_sample
], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

final_df.to_csv('Balanced_DynaHate_Sample.csv', index=False)