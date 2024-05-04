import pymorphy3
morph = pymorphy3.MorphAnalyzer()

def lemmatize(text):
    words = text.split() # разбиваем текст на слова
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)

    return ' '.join(res)

map_bank = {'Тинькофф': 'tink',
                'Сбербанк': 'sber',
                'ВТБ': 'vtb',
                'Альфа-банк': 'alpha'}

# map_bank_eng_to_rus = {'tink': 'Тинькофф',
#                 'sber': 'Сбербанк',
#                 'vtb': 'ВТБ',
#                 'alpha': 'Альфа-банк'}

map_bank_reverse = dict(zip(map_bank.values(), map_bank.keys()))
map_bank_eng_to_rus = map_bank_reverse


colors = {'tink': 'yellow',
          'alpha': 'red',
          'vtb': 'blue',
          'sber': 'green',
         }


map_category = {
                'debitCards': 'Дебетовые карты',
                'creditCards': 'Кредитные карты',
                'remoteService': 'Дистанционное обслуживание',
                'serviceLevel': 'Обслуживание',
                'credits': 'Кредиты наличными',
                'savings': 'Вклады',
                'mortgage': 'Ипотека',
                'other': 'Другое'
               }


stop_words = ['банк', 'деньга', 'отзыв']
stop_words_patterns = ['банк']


def is_stop_word(x):
    if x in stop_words:
        return True
    for pattern in stop_words_patterns:
        if pattern in x:
            return True
        
    return False



def generate_axis(nrows, ncols, n_competitors, axes):
    n = 10
    x, y = -1, 0    
    
    for i in range(n_competitors+1):
        x += 1
        if x == nrows:
            y += 1
            x = 0
        if nrows == 1:
            ax = axes[y]
        else:
            ax = axes[x][y]

        yield ax
        

    