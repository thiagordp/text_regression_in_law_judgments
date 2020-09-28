"""
Remove the result of each process

@date       26/03/2020
@author     Thiago Raulino Dal Pont
"""
import glob

from tqdm import tqdm

PATH_SRC_DATASET = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/" \
                   "Datasets/processos_transp_aereo/merge_com_dispositivo/"

PATH_DST_DATASET = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/" \
                   "Datasets/processos_transp_aereo/merge_sem_dispositivo/"

PATH_LAB_PROC = "procedente/"
PATH_LAB_INPROC = "improcedente/"
PATH_LAB_PARC_PROC = "parcialmente_procedente/"
PATH_LAB_EXT = "extincao/"

PATH_LAB = [PATH_LAB_PROC, PATH_LAB_INPROC, PATH_LAB_PARC_PROC, PATH_LAB_EXT]

FORBIDDEN_WORDS = ["dispositivo", "ante o exposto", "diante do exposto"]
ALLOWED_WORDS = ["dispositivos", "dispositivo legal", "dispositivos legais", "dispositivo do", "dispositivo citado",
                 "dispositivo:", "dispositivo revela", "diante o exposto não", "mesmo dispositivo", "o dispositivo"]
count = 0


def process_file(src_file_path, dst_file_path):
    forbidden_flag = 0
    global count
    with open(src_file_path, "r") as src_file:
        with open(dst_file_path, "w+") as dst_file:
            for src_line in src_file.readlines():

                has_allowed = 0

                for word in ALLOWED_WORDS:
                    if src_line.lower().find(word) != -1:
                        has_allowed = 1
                        break

                for word in FORBIDDEN_WORDS:
                    if src_line.lower().find(word) != -1 and has_allowed == 0:
                        forbidden_flag = 1

                count += forbidden_flag

                token = src_line.strip().split()

                if len(token) > 0 and token[0].find("Florianópolis") != -1:
                    forbidden_flag = 0
                    dst_file.write("\n")

                if forbidden_flag == 0:
                    dst_file.write(src_line)


if __name__ == "__main__":
    print("Remove result of each process")

    for path_type in PATH_LAB:
        src_path = PATH_SRC_DATASET + path_type
        dst_path = PATH_DST_DATASET + path_type

        files = glob.glob(src_path + "*.txt")

        for path in files:
            dst_f_path = path.replace(src_path, dst_path)
            process_file(path, dst_f_path)

    print(count)
