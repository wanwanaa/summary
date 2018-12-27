from rouge import FilesRouge


def rouge_score(filename_gold, filename_result):
    files_rouge = FilesRouge(filename_gold, filename_result)
    scores = files_rouge.get_scores(avg=True)
    return scores


def write_rouge(filename, score):
    rouge_1 = 'ROUGE-1 f ' + str(score['rouge-1']['f']) + ' ' \
              + str(score['rouge-1']['p']) + ' ' \
              + str(score['rouge-1']['r'])
    rouge_2 = 'ROUGE-2 f ' + str(score['rouge-2']['f']) + ' ' \
              + str(score['rouge-2']['p']) + ' ' \
              + str(score['rouge-2']['r'])
    rouge_l = 'ROUGE-l f ' + str(score['rouge-l']['f']) + ' ' \
              + str(score['rouge-l']['p']) + ' ' \
              + str(score['rouge-l']['r'])
    rouge = [rouge_1, rouge_2, rouge_l]
    with open(filename, 'w') as f:
        f.write('\n'.join(rouge))
