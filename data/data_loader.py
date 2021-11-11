import emoji
import nltk
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


# TextPreProcessor preprocesses the text used
text_processor = TextPreProcessor(
    # these terms will be normalized by TextPreProcessor
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # these terms will be annotated by TextPreProcessor 
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    #TextPreProcessor fix HTML tokens
    fix_html=True,  
    # corpus from which the word statistics are going to be use. for word segmentation
    segmenter="twitter",
    # the corpus from which the word statistics are going to be used. for spell correction
    corrector="twitter",
    # make TextPreProcessor perform word segmentation on hashtags
    unpack_hashtags=True,  
    # make  Unpack contractions (can't -> can not)
    unpack_contractions=True,
    # spell correction for elongated words  
    spell_correct_elong=False,  
    # select a tokenizer. We use SocialTokenizer
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    # emoticons is a  list of dictionaries, for replacing tokens extracted from the text with other expressions
    dicts=[emoticons]
)


def delete_emoji_underscores(text):
    tokens = text.split()
    res = []
    for token in tokens:
        if len(token) > 3 and '_' in token:
            token = token.replace('_', ' ')

        if token[0] == '<' and token[-1] == '>':
            token = token[1:-1]

        res.append(token)
    return ' '.join(res)


def process(text):
    text = text.lower().strip()
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = ' '.join(text_processor.pre_process_doc(text))
    text = delete_emoji_underscores(text)
    return text

 # data_path = 'data/train.txt'
def load_data_context(data_path='data/train.txt', is_train=True):
    E_DICT = {'happy': 0,
                'angry': 1,
                'sad': 2,
                'others': 3}
    d_list = []
    t_list = []
    f = open(data_path, 'r')
    data_lines = f.readlines()
    f.close()
    for i, text in enumerate(data_lines):
        # we skip the first line
        if i == 0:
            continue
        #split at each tab character
        tokens = text.split('\t')
        convers = tokens[1:4]

        a = convers[0]
        b = convers[1]
        c = convers[2]

        a = process(a)
        b = process(b)
        c = process(c)

        d_list.append(a + b + c)
        if is_train:
            emo = tokens[3 + 1].strip()
            t_list.append(E_DICT[emo])

    if is_train:
        return d_list, t_list
    else:
        return d_list

