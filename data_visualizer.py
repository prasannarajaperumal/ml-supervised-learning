import pandas as pd
import matplotlib.pyplot as plt

col_names = ['Employment', 'Education', 'Marital Status', 'Position', 'Family Status', 'Ethnicity', 'Sex', 'Country', 'Salary']
df = pd.read_csv('data/adult-salary-data.csv', header=None, names=col_names)
df.head(1)
df["Marital Status"].value_counts().plot.bar()
plt.show()


# for col_name in col_names:
#     df[col_name].plot(kind='hist')
