
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
sns.__version__
#%%

def plot_model(y_true, y_pred):

    fig, ax = plt.subplots(ncols=2, nrows=2, constrained_layout=True, figsize=(10,8))

    all_y = y_true.to_list() + y_pred.tolist()
    set_y = ["True Value"]*len(y_true) + ["Predictive Value"]*len(y_pred)
    ax[0,0].set_title('True Value vs Predictive Value Distribution')
    ax[0,0].set_xlabel('y')
    sns.histplot(x=all_y, hue=set_y, kde=True, ax=ax[0,0])
    

    ax[0,1].set_title('True Value vs Predictive Value')
    ax[0,1].set_xlabel('y true')
    ax[0,1].set_ylabel('y predictive')
    sns.scatterplot(x=y_true, y=y_pred, ax=ax[0,1])
    limite = np.floor(np.max([y_true, y_pred])) + 1
    ax[0,1].plot([0,limite], [0,limite], color='r')


    ax[1,0].set_title('Residuals Distribution')
    ax[1,0].set_xlabel("Residuals")
    sns.histplot((y_true-y_pred), ax=ax[1,0], kde=True)

    ax[1,1].set_xlabel("y predictive")
    ax[1,1].set_ylabel("Residuals")
    ax[1,1].set_title('Residuals vs Predictive Value')
    sns.scatterplot(x=y_pred, y=(y_true-y_pred), ax=ax[1,1])