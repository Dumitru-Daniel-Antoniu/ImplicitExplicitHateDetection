import pandas as pd

dh_df = pd.read_csv("C:\\Users\\ddumi\\Desktop\\Faculty\\MasterFirstYear\\SecondSemester\\StatisticalProcessingOfNaturalLanguage\\ProjectResources\\DynaHate.csv")

hate_train = dh_df[(dh_df['label'] == 'hate') & (dh_df['split'] == 'train')]
nothate_train = dh_df[(dh_df['label'] == 'nothate') & (dh_df['split'] == 'train')]

hate_train_sample = hate_train.sample(n=100, random_state=42)
nothate_train_sample = nothate_train.sample(n=100, random_state=42)

cg_df = pd.read_csv("ChatGpt_comments_train.csv")

final_df = pd.concat([
    hate_train_sample,
    nothate_train_sample,
    cg_df
], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

final_df.to_csv('ChatGpt_DynaHate_Sample.csv', index=False)