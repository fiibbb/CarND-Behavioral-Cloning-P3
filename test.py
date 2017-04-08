import csv

with open('recording/01/driving_log.csv') as f:
    reader = csv.reader(f)
    angles = [line[3] for line in reader][1:]
    angles.sort()
    print(angles[:-500])
