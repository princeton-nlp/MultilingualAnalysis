import lang2vec.lang2vec as l2v
from scipy import spatial

source_langs = ['en']
target_langs = ['fr', 'hi', 'ar']

# feature = 'geo'
feature = 'syntax_wals'

for s_lang in source_langs:
    for t_lang in target_langs:
        s_features = l2v.get_features(s_lang, feature)[s_lang]
        t_features = l2v.get_features(t_lang, feature)[t_lang]
        if isinstance(s_features, dict):
            s_features = s_features[s_lang]
            t_features = t_features[t_lang]

        for i in range(len(s_features)):
            if s_features[i] == '--':
                s_features[i] = 0.0
            if t_features[i] == '--':
                t_features[i] = 0.0
 
        cos_sim = spatial.distance.cosine(s_features, t_features)
        print("{} {} {}".format(s_lang, t_lang, cos_sim)) 
