from Ai_exec import create_ai
from colab2 import use_model_and_predict
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run AI creation script with specific parameters. --train, --safe, --validate and --predict default to False and can be set to True.")
    parser.add_argument('--filepath', type=str, required=False, default='./data/Dataset.xlsx',
                        help='Der Pfad zur Datendatei.')
    parser.add_argument('--output_path', type=str, required=False, default='./output',
                        help='Der Pfad zur Datendatei.')
    parser.add_argument('--save_file', type=str, required=False, default="./AI/EMS2_AI",
                        help='Der Speicherort für die Ausgabe-Datei.')
    parser.add_argument('--train', action='store_true', help='Aktiviert das Training.')
    parser.add_argument('--save', action='store_true', help='Aktiviert das sichere Speichern.')
    parser.add_argument('--validate', action='store_true', help='Aktiviert die Validierung.')
    parser.add_argument('--predict', action='store_true', help='Aktiviert die Vorhersage.')
    parser.add_argument('--old', action='store_true', help='Verwende das alte Model.')

    args = parser.parse_args()

    # Verwende den Standardpfad, wenn --filepath nicht angegeben ist
    create_ai(filepath=args.filepath,
              output_file=args.output_path,
              save_file=args.save_file,
              train=args.train,
              safe=args.save,
              validate=args.validate,
              predict=args.predict,
              old=args.old)

