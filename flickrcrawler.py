import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import flickrapi
import urllib
from scipy import misc
from xml.etree import ElementTree
import requests
import multiprocessing

def crawl(keyword):
    flickr=flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)
    #keywords = ['outside', 'morning', 'afternoon', 'evening', 'night', 'city', 'raining', 'winter', 'summer', 'autumn', 'fall', 'building', 'farm']
    # keywords = ['farm']

    tag_mode = 'all'

    photos = flickr.walk(text=keyword,
                         tag_mode=tag_mode,
                         tags=keyword,
                         extras='url_c',
                         per_page=100,           # may be you can try different numbers..
                         sort='relevance')

    n = 0

    while (photo := next(photos, None)) is not None:
        try:
            url = photo.get('url_c')
            if url is not None:
                dt = None
                for tag in flickr.photos_getExif(photo_id=photo.get('id')).iter('exif'): #getiterator('exif'):
                    if tag.attrib['tag'] == 'DateTimeOriginal':
                        dt = list(tag)[0].text.strip().encode('utf-8')
                        break
                if dt is None:
                    print("No datetime in exif")
                else:
                    photo_id = photo.get('id')
                    img_data = requests.get(url).content
                    with open('images/'+str(photo_id)+'.jpg', 'wb+') as f:
                        f.write(img_data)
                    with open('images_meta/'+str(photo_id)+'.txt', 'wb+') as f:
                        f.write(dt)
                    n+=1
                    if n%20 == 0:
                        print(str(keyword) + ': ' + str(n) + ' images in total')
            else:
                print('no url found')
        except flickrapi.FlickrError:
            print('caught error')

if __name__ == '__main__':
    jobs = []
    # keywords = ['outside', 'morning', 'afternoon', 'evening', 'night', 'city', 'raining', 'winter', 'summer', 'autumn', 'fall', 'building', 'farm', 'street']
    # keywords = ['beach', 'castle', 'valley', 'mountain', 'wildlife', 'creek', 'lake', 'forest', 'river']
    # keywords = ['skyline', 'desert', 'harbour', 'village', 'hill']
    keywords = ['airport', 'park']
    for i in keywords:
        p = multiprocessing.Process(target=crawl, args=(i,))
        jobs.append(p)
        p.start()