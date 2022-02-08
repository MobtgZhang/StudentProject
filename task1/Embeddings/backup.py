                # START
                if '<START>' not in word_frequency:
                    word_frequency[data_dict['<START>']] = 1
                else:
                    word_frequency[data_dict['<START>']] += 1
                # END
                if '<END>' not in word_frequency:
                    word_frequency[data_dict['<END>']] = 1
                else:
                    word_frequency[data_dict['<END>']] += 1
                tp_length = len(sentence)
                remain_length = max_length+2-tp_length
                if '<PAD>' not in word_frequency:
                    word_frequency[data_dict['<PAD>']] = remain_length
                else:
                    word_frequency[data_dict['<PAD>']] += remain_length
                if len(sentence)<256:
                    sentence = sentence[]
                else:
                    print(sentence)
                # PAD
                for _ in range(remain_length):
                    sentence.append('<PAD>')
                sentence = ['<START>'] + sentence
                sentence = sentence + ['<END>']