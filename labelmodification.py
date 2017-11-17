import json
from sklearn.preprocessing import scale
with open("output.txt") as data:
    subjects = json.load(data)


all_y = [int(subject[2]) for subject in subjects]
scale_y = scale(all_y)

def labelmodifications():
    x1 = []
    x2 = []
    linear = []
    binary = []
    for i,e in enumerate(all_y):
        x1.append(subjects[i][3])
        x2.append(subjects[i][4])
        linear.append(scale_y[i])
        if e <= 1:
            binary.append(0)       
        elif e >= 3:
            binary.append(1)
    
    return [x1, x2, all_y, linear, binary]


new_data = labelmodifications()
print(sorted(scale_y))

