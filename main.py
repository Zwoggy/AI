from Ai_exec import create_ai
from colab2 import use_model_and_predict

if __name__ == '__main__':

    create_ai(filepath='C:/Users/fkori/PycharmProjects/AI/data/Dataset-without-1550.xlsx',
              save_file="",
              train=False,
              safe=False,
              validate=False,
              predict=False)
