import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('analysis/model_evaluation_detailed.csv')

# ... [Your loop code that populates detailed_df] ...

# Once all metrics are collected, create the box plot
import matplotlib.pyplot as plt
import seaborn as sns

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=8, ncols=2, figsize = (12,24))

for i, hyperparameter in enumerate(['ImageInput','Augmentation','Scheduler','CriterionWeight','OptimizerType','OptimizerLR','LossFunction','Model']):
    for j, metric in enumerate(['Accuracy','AUC']):  
        sns.boxplot(x=hyperparameter, y=metric, data=df, ax=axes[i, j])
        axes[i, j].set_title(f"{metric} Comparison by {hyperparameter}")
        axes[i, j].set_xlabel(f"{hyperparameter}")
        axes[i, j].set_ylabel(f"{metric}")
        

# Adjust layout for clarity
plt.tight_layout()
plt.show()


