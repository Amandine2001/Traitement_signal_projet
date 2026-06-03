import bandeSonore
import pandas as pd
 
 
def create_dataframe(
    features_list: list[dict[str, bandeSonore.bandeSonore | str]],
):
 
    data = list()
 
    for item in features_list:
        audio: bandeSonore.bandeSonore = item["audio"]
        mecanisme: str = item["mecanisme"]
        modele: str = item["modele"]
        normalite: str = item["normalite"]
 
        data.append(
            {
                "name": audio.name,
                "mecanisme": mecanisme,
                "modele": modele,
                "normalite": normalite,
                "centroid": audio.centroid,
                "mfcc_mean": audio.mfcc.mean(),
                "chroma_mean": audio.chroma.mean(),
                "tempo": audio.tempo,
                "zcr": audio.zcr,
            }
        )
 
    df = pd.DataFrame(data)
    return df