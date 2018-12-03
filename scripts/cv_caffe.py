import numpy as np
import os


def cv(infile, outfile, m, base):
    try:
        bb = np.loadtxt(infile)
    except:
        pass
    f = open(outfile, 'w')
    m = 10
    for b in bb:
        k = int(b[0]) + (m - 0)
        if k % m == 0:
            w = b[4] - b[2]
            h = b[5] - b[3]
            xc = b[2] + w / 2
            yc = b[3] + h / 2

            f.write('{}{:04d}.png {} {} {} {} {}\n'.format(base, k//m, b[1], xc/1280, yc/964, w/1280, h/964))
    f.close()


directory = 'r22'
m = {
    'video0001': 10,
    'video0002': 10,
    'v8ss': 25,
    'zahrada4-eq': 10,
    '2017091908_tul': 10,
    '2017120812_2.2m_radial': 2,
    '2017120812_2.2m_tangens': 2,
    '2017120812_4m_radial': 2,
    '2017120812_4m_tangens': 2,
}

for f in os.listdir(directory):
    bf = f.rsplit('.', 1)[0]
    print(bf)

    base = '/home/honza/fastssd/data/_videa/{}/SrcImages/img_'.format(bf)
    cv(os.path.join(directory, f), directory + '_' + f, m[bf], base)
