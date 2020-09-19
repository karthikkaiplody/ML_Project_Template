import pandas as pd 
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    # Creating the dummy column
    df['kfold'] = -1

    # Shuffle the data and reseting the indices
    df = df.sample(frac=1).reset_index(drop=True)

    # Creating the KFolds
    kf = StratifiedKFold(n_splits=5, shuffle=False)

    for fold , (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv('input/train_folds.csv', index=False) 

