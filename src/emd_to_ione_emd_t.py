f=open('T2.txt')
fw=open('twitter.embedding.update.2SameAnchor.1.foldtrain.twodirectionContext.number.100_dim.10000000','w')
for i in f:
    ii=i.split()
    strt=ii[0]+'_twitter'+' '
    for iii in ii[1:]:
        strt=strt+iii+'|'
    fw.write(strt+'\n')