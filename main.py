from Ai_exec import create_ai
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
    parser.add_argument('--validate_45_Blind', action='store_true', help='Aktiviert die Validierung.')
    parser.add_argument('--validate_BP3C', action='store_true', help='Aktiviert die Validierung.')
    parser.add_argument('--predict', action='store_true', help='Aktiviert die Vorhersage.')
    parser.add_argument('--old', action='store_true', help='Verwende das alte Model.')
    parser.add_argument('--gpu_split', action='store_true', help='Teile das Model auf 4 GPUs auf.')
    parser.add_argument('--big_dataset', action='store_true', help='Verwende den größeren Datensatz.')
    parser.add_argument('--use_structure', action='store_true', help='Verwende die von Alphafold 2 vorhergesagten Strukturen.')
    args = parser.parse_args()

    # Verwende den Standardpfad, wenn --filepath nicht angegeben ist
    create_ai(filepath=args.filepath,
              output_file=args.output_path,
              save_file=args.save_file,
              train=args.train,
              safe=args.save,
              validate_45_Blind=args.validate_45_Blind,
              validate_BP3C=args.validate_BP3C,
              predict=args.predict,
              old=args.old,
              gpu_split=args.gpu_split,
              big_dataset=args.big_dataset,
              use_structure=args.use_structure)

