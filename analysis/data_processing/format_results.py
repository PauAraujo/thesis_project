def process_results(text, task):
    lines = text.split('\n')  # split the text into lines
    if task == 'sib':

        # extract the accuracy numbers and round them to 4 d.p.
        accuracy_numbers = [round(float(lines[i]), 4) for i in range(1, len(lines), 2)]
        # format the numbers
        formatted_output = '[\n    '
        for i in range(0, len(accuracy_numbers), 10):
            formatted_output += ', '.join(map(str, accuracy_numbers[i:i + 10]))
            if i + 10 < len(accuracy_numbers):
                formatted_output += ',\n    '
            else:
                formatted_output += '\n'
        formatted_output += ']'
        # print formatted numbers, to have 10 per line
        print(formatted_output)
    else:
        # extract the accuracy numbers and round them to 4 d.p.
        accuracy_numbers = [round(float(lines[i]), 4) for i in range(1, len(lines), 2)]
        # print the numbers in one line
        print(accuracy_numbers)


def extract_language_accuracy_pairs(text, task):
    lines = text.split('\n')
    if task=='sib':
        collect = False # flag to start collecting when 'deuLatn' is found
        language_accuracy_pairs = []
        for i in range(0, len(lines), 2):
            # get the language code part (e.g., 'engLatn')
            lang_code = lines[i].split('_')[-1]
            # get the accuracy and round it to 4 d.p.
            accuracy = round(float(lines[i + 1]), 4)
            if lang_code == 'deuLatn':
                collect = True  # START collecting from 'deuLatn'
            if collect:
                language_accuracy_pairs.append((lang_code, accuracy))
            if lang_code == 'fraLatn':
                break  # STOP collecting after 'fraLatn'
    else:
        # extract the language codes and accuracy numbers, then round the accuracy numbers to 4 d.p.
        language_accuracy_pairs = [(lines[i].split('_')[-1], round(float(lines[i + 1]), 4)) for i in range(0, len(lines), 2)]
        # print each language code and accuracy number pair
    for lang, accuracy in language_accuracy_pairs:
        print(f"{lang}\t{accuracy}")



if __name__ == "__main__":
    from accuracy_string import text
    task = "sib"
    extract_language_accuracy_pairs(text,task)
    process_results(text,task)
