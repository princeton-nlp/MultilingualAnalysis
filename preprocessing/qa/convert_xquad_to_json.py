import argparse
from tqdm import tqdm
import os
import csv
import json
import newlinejson as nlj

def convert_dataset_to_json(args):
    # File to write in
    json_write = open(os.path.join(args.save_dir, os.path.split(args.file)[1]), 'w')
    all_json_objs = []
    with open(args.file) as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "").strip()
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    question = qa["question"].strip()
                    id_ = qa["id"]

                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"].strip() for answer in qa["answers"]]

                    # Features currently used are "context", "question", and "answers".
                    # Others are extracted here for the ease of future expansions.
                    # Construct dictionary and write
                    single_question = {
                        "title": title,
                        "context": context,
                        "question": question,
                        "id": id_,
                        "answers": {
                            "answer_start": answer_starts,
                            "text": answers,
                        },
                    }
                    # all_json_objs.append(single_question)
                    # Write
                    json_write.write(json.dumps(single_question)+'\n')

    # json_write.write(json.dumps(all_json_objs))

def main():
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument("--file", required=True, type=str, help="Path to file containing the corpus.")
    parser.add_argument("--save_dir", default=None, type=str, help="Path to directory where file will be saved.")

    args = parser.parse_args()

    # Convert documents to CONLLU files
    convert_dataset_to_json(args)

if __name__ == '__main__':
    main()