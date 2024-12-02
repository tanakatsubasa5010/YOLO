import re

data_dict = {'P': 2, 'R': 3,'AP50': 4, 'AP75': 5, 'mAP': 6}
with open('val.txt', 'r') as f:
    f = f.readlines()

data = []
for f_ in f:
    if f_[:11] != 'Ultralytics' and f_[:13] != 'val: Scanning' and f_[:16] != 'Results saved to' and f_[:44] != '                 Class     Images  Instances':
        if f_[:6] == 'Speed:':
            f_ = f_.split(' ')
            time = float(f_[1][:-2]) + float(f_[3][:-2]) + float(f_[7][:-2])
            data[-1][1:1] = [str(time)]
        elif f_[0] == ' ':
            d = re.findall('\s*([^\s]+)\s*', f_)
            data[-1] += d[3:]
        else:
            data.append([])
            data[-1].append(f_.split(' ')[0].lower())

with open('out.csv', 'w') as g:
    g.write('\n'.join(list(map(lambda x: ','.join(x), data))))
    print(data)
#     for f_ in f:
#         match_ = re.findall('.*?(\d+\.?\d*).*?', f_)
#         data.append([float(i) for i in match_])
#     for i in [data_dict['P'], data_dict['R'], data_dict['AP50'], data_dict['AP75'], data_dict['mAP']]:
#         for j in data:
#             print(j[i], end=',')
#         print()