# PSML
## paper: Towards Improving Embedding Based Models of Social Network Alignment via Pseudo Anchors

PSML_IONE,PSML_ABNE,PSML_DEEPLINK,PSML_SNNA:<br>

```
numpy==1.14<br>
networkx==2.0<br>
scipy==0.19.1<br>
tensorflow>=1.12.1<br>
gensim==3.0.1<br>
scikit-learn==0.19.0<br>
```

PSML_DALUAP,PSML_MGCN:<br>
```
python >= 3.6<br>
pytorch >= 0.4<br>
numpy  1.18.0<br>
tqdm<br>
networkx >2.0<br>
```

support data comes from :https://github.com/ChuXiaokai/CrossMNA<br>
query data comes from :https://github.com/ColaLL/IONE<br> , https://github.com/ColaLL/AcrossNetworkEmbeddingDiversity

### For PSML_IONE<br>
```
   first run PSML_IONE.py
   second run Four.py
   getPrecision--should run emd_to_ione_emd.py and emd_to_ione_emd_t.py
```
### For PSML_ABNE<br>
```
   first run PSML_ABNE.py
   second run Four.py
   getPrecision--should run emd_to_ione_emd.py and emd_to_ione_emd_t.py
```
### For PSML_SNNA<br>
```
   use deepwalk or line get pre_data
   run PSML_SNNA.py
```
### For PSML_DeepLink<br>
```
   run embedding.py to use word2vec get pre_data
   run PSML_Deeplink.py
```
### For PSML_MGCN<br>
```
   run PSML_MGCN.py
```
### For PSML_DALUAP<br>
```
   run PSML_DALUAP.py
```
## NOTE:
Method of adding pseudo node, Take two pseudo anchors, which are connected to each other, such as subnetwork file:
```
        node      node
         1          2
         3(anchor)  4
```
You need change it to:
```
        node      node
         1          2
         3(anchor)  4
         3          5(pse)
         3          6(pse)
         5(pse)     3
         6(pse)     3
         5(pse)     6(pse)
         6(pse)     5(pse)
         
```

### other
Some students send email asked me if I had the tensorflow version of ione/abne. Here is the version I reproduced:
https://github.com/yanzihan1/IONE-Aligning-Users-across-Social-Networks-Using-Network-Embedding
### contact me:
yzhcqupt@163.com


