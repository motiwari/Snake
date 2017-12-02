import pickle
import csv
def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file,  dialect='excel',delimiter='|')
        for line in data:
            writer.writerow([line])
#----------------------------------------------------------------------
if __name__ == "__main__":
    data = ["first_name,last_name,city".split(","),
            "Tyrese,Hirthe,Strackeport".split(","),
            "Jules,Dicki,Lake Nickolasville".split(","),
            "Dedric,Medhurst,Stiedemannberg".split(",")
            ]
    replay_memory = pickle.load(open("finalscores.pkl","rb"))
    path = "output.csv"
    csv_writer(replay_memory, path)
