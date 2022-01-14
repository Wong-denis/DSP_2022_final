from audio import AudioFeature
from model import Model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import pickle


def parse_audio_playlist(playlist):
    """
    Assumes an Apple Music playlist saved as plain text as parse input.
    Returns: zip object with (paths, genres)
    """

    df = pd.read_csv(playlist, sep=" \t")
    #print(df[["Genre"]])
    if df[["Genre"]].isnull().values.any():
        print('\nlist has empty genre!!!\n')
        df = df[["Location"]]
        paths = df["Location"].values.astype(str)
        return paths
    #print(type(df))
    df = df[["Location", "Genre"]]
    

    paths = df["Location"].values.astype(str)
    #paths = np.char.replace(paths, "Macintosh HD", "")

    genres = df["Genre"].values

    return zip(paths, genres)
        

def load_saved_audio_features(path):
    
    files = next(os.walk(path))
    pkl_list = files[-1]
    pkl_list = [p for p in pkl_list if p.endswith(".pkl")]
    
    audio_features = []
    
    for p in pkl_list:
        with open(f"{path}{p}", "rb") as input_file:
            e = pickle.load(input_file)
            audio_features.append(e)
    
    return audio_features

def cout_genre_data(genre,cin_path,cout_path):
    df = pd.read_csv(cin_path, sep=" \t")
    df = df.drop("Genre", 1)
    df["Genre"] = genre
    df.to_csv(cout_path,index=False,sep='\t')

if __name__ == "__main__":

    all_metadata = parse_audio_playlist(playlist="data/Subset.txt")
    classify_data = parse_audio_playlist(playlist="data/Need_classify.txt")
    audio_cd_features = []
    for md in classify_data:
        path= md
        audio_cd = AudioFeature(path)
        audio_cd.extract_features("mfcc", "spectral_contrast", "tempo", save_local=True)
        audio_cd_features.append(audio_cd)

    #print(audio_cd_features)

    audio_features = []
    for metadata in all_metadata:
        path, genre = metadata
        audio = AudioFeature(path, genre)
        audio.extract_features("mfcc", "spectral_contrast", "tempo", save_local=True)
        audio_features.append(audio)
    #print(audio_features)
    # audio_features = load_saved_audio_features("./data/")
    
    feature_matrix = np.vstack([audio.features for audio in audio_features])
    cd_feature_matrix = np.vstack([audio_cd.features for audio_cd in audio_cd_features])
    genre_labels = [audio.genre for audio in audio_features]

    model_cfg = dict(
        tt_test_dict=dict(shuffle=True, test_size=0.3),
        tt_val_dict=dict(shuffle=True, test_size=0.25),
        scaler=StandardScaler(copy=True),
        base_model=RandomForestClassifier(
            random_state=42,
            n_jobs=4,
            class_weight="balanced",
            n_estimators=250,
            bootstrap=True,
        ),
        param_grid=dict(
            model__criterion=["entropy", "gini"],
            model__max_features=["log2", "sqrt"],
            model__min_samples_leaf=np.arange(2, 4),
        ),
        grid_dict=dict(n_jobs=4, refit=True, iid=False, scoring="balanced_accuracy"),
        kf_dict=dict(n_splits=3, random_state=42, shuffle=True),
    )
    model_cd = Model(cd_feature_matrix,genre_labels,model_cfg)
    model = Model(feature_matrix, genre_labels, model_cfg)
    model.train_kfold()
    model.predict(holdout_type="val")
    model.predict(holdout_type="test")
    model_cd.best_estimator = model.best_estimator
    prediction = model_cd._predict(holdout_type="classify")
    print(prediction)
    cout_genre_data(prediction,"data/Need_classify.txt","classify.csv")


    