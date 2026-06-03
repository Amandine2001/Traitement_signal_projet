import os
def set_fata():
    data = []
    data_path= "audio"
    for filename in os.listdir(data_path):
    
    label = nom_labels.get(filename, 'inconnu')
    for son in os.listdir(os.path.join(data_path, filename)):
        if son.endswith('.wav'):
            dataset.append((os.path.join(data_path, filename, son), label))