class Config():
    def __init__(self):
        # dataset
        self.filename_train = 'DATA/LCSTS/PART_I.txt'
        self.filename_valid = 'DATA/LCSTS/PART_II.txt'
        self.filename_test = 'DATA/LCSTS/PART_III.txt'

        # glove
        self.dim = 300
        self.filename_glove = 'DATA/glove.6B/glove.6B.{}d.txt'.format(self.dim)

        # trimmed data
        self.filename_trimmed_train_text = 'DATA/result/train_text.pt'
        self.filename_trimmed_train_summary = 'DATA/result/train_summary.pt'

        self.filename_trimmed_valid_text = 'DATA/result/valid_text.pt'
        self.filename_trimmed_valid_summary = 'DATA/result/valid_summary.pt'

        self.filename_trimmed_test_text = 'DATA/result/test_text.pt'
        self.filename_trimmed_test_summary = 'DATA/result/test_summary.pt'

        # embedding
        self.filename_embeddings = 'DATA/result/glove_embeddings_{}d.pt'.format(self.dim)

        # sequence length
        self.seq_len = 114
        self.summary_len = 22

        # bos
        self.bos = 2

        # vocab
        self.filename_words = 'DATA/result/word2index.pkl'
        self.filename_index = 'DATA/result/index2word.pkl'
        self.vocab_size = 4000

        # gold summaries
        self.gold_summaries = 'DATA/result/gold/gold_summaries.txt'