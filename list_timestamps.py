import sys,os
dirname = os.path.dirname(__file__)
directory = os.path.join(dirname, 'images_meta')

buckets = [0]*24

for filename in os.listdir('images_meta'):
    with open('images_meta/'+filename, 'r') as f:
        h = f.read().split(' ')[1].split(':')[0]
        buckets[int(h)] += 1

print(buckets)
