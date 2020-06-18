"""

"""
import glob

PATH_DATASET = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/processos_transp_aereo/merge_sem_dispositivo/"

def list_judgement_class():
    folders = glob.glob(PATH_DATASET + "*")

    for folder in folders:
        # print(folder.replace(PATH_DATASET, ""))

        files = glob.glob(folder + "/*.txt")
        # print(len(files))

        folder = folder.replace(PATH_DATASET, "")
        folder_name = folder.replace("extincao", "E")
        folder_name = folder_name.replace("improcedente", "I")
        folder_name = folder_name.replace("parcialmente_procedente", "PP")
        folder_name = folder_name.replace("procedente", "PT")

        for f in files:
            file_name = f.replace(PATH_DATASET, "").replace(".txt", "").replace(folder, "").replace("/", "")

            print(file_name, folder_name, sep="\t")


def main():
    return None


if __name__ == "__main__":

    main()
