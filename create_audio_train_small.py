import pandas as pd
s1=pd.read_csv('data/audio/train.csv')
s2=pd.read_csv('data/audio/augmented/augmented_audio_dataset.csv')
small=pd.concat([s1,s2],ignore_index=True).sample(n=20,random_state=42)
small.to_csv('data/audio/train_small.csv',index=False)
print('Wrote small CSV rows=',len(small))