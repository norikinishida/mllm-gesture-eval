import argparse
import json
import re


def main(args):

    type2total = {}
    type2score = {}

    for typ in ["Indexical", "Iconic", "Symbolic", "Overall"]:
        type2total[typ] = 0
        type2score[typ] = 0.0

    with open(args.input) as f:
        for line in f.readlines():
            example = json.loads(line.strip())

            # score = example["evaluation"].replace("\n", " ").split(" ")
            # score = float(score[1])
            match = re.search(r"(評価スコア|Score)\s*[:：]\s*([0-9]*\.?[0-9]+)", example["evaluation"])
            if match is not None:
                score = float(match.group(2))

                type2total["Overall"] += 1
                type2score["Overall"] += score

                type2total[example["gesture_type"]] += 1
                type2score[example["gesture_type"]] += score

            else:
                print(match)

    for typ in ["Indexical", "Iconic", "Symbolic", "Overall"]:
        type2score[typ] /= type2total[typ]

    print(type2total)
    print(type2score)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    main(args)

