import json
import os


def cv_to_json(infile):
    with open(infile, 'r') as f:
        annots = {}
        for line in f:
            v = line.split(' ')

            image = v[0]
            dd = image.split("/")
            iname = "_".join(dd[-3:])
            # iname = image[image.find('_videa') + 7:]
            # iname = iname.replace('/', '_')

            if iname not in annots:
                annots[iname] = {}
                annots[iname]['boxes'] = []
                annots[iname]['scores'] = []

            prob = float(v[1])
            prob = int(round(prob * 1000))
            x, y, w, h = [float(n) for n in v[2:]]

            x1 = x - w / 2
            x2 = x + w / 2
            y1 = y - h / 2
            y2 = y + h / 2

            if prob > 10:
                annots[iname]['boxes'].append([x1, y1, x2, y2])
                annots[iname]['scores'].append(prob)

    outfile = os.path.basename(infile)
    with open(outfile.rsplit('.')[0] + '.json', 'w') as of:
        json.dump(annots, of)

    videa = ['video0001',
             'video0002',
             'v8ss',
             'zahrada4',
             '2017091908_tul',
             '2017120812_2.2m_radial',
             '2017120812_2.2m_tangens',
             'difficult',
             'chairs']

    for v in videa:
        dv = {k: annots[k] for k in annots if v in k}
        if len(dv) > 0:
            with open(outfile.split('.')[0] + '_' + v + '.json', 'w') as of:
                json.dump(dv, of)


infiles = [
'../r22cv1.txt',
]

for infile in infiles:
    cv_to_json(os.path.join('./txt', infile))
