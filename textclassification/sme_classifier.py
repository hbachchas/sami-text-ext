import fasttext

def language_identifier(model, text):
    """
    Identify language
    """
    # Predict the language
    return model.predict(text, k=1)


def sme_classifier(model, text):
    """
    Sami classifier.
    """

    pred = language_identifier(model, text)

    if pred[1][0] < 0.4:
        return "__label__sme" # Sami
    elif pred[1][0] > 0.6:
        return pred[0][0]   # Another language
    else:
        return "__label__unclassified" # Unidentified -> Might be sami, might be unfamiliar language (ex. Swahili), or a mix of languages (ex. norwegian + sami)

def test_classifier():
    """
    This method will test the classifier lid.176 on a sme_label.txt.
    __label__sme -> sami text
    __label__nos -> no sami text
    __label__unclassified -> unidentified text. Might be sami, or another language the lid.176 is not familiar with (ex. Swahili). I can also indicate a mix of languages.
    """

    model_path = 'lid.176.bin'
    model = fasttext.load_model(model_path)

    with open("shuf_sme_label.txt", "r") as file:
        data = file.readlines()
    
    hit = 0
    total = 0
    unclassified = 0

    for line in data:
        
        class_ = sme_classifier(model, line[13:-1])
        
        if class_ == "__label__sme":
            if line[0:12] == "__label__sme":
                hit += 1
        elif class_ == "__label__unclassified":
            unclassified += 1
        else:
            if line[0:12] == "__label__nos":
                hit += 1
        total += 1

    print("Accuracy: " + str(hit/total))
        


def main():

    test_classifier()

    # RUN sme_classifier(model, text) to classify sami text
    #
    # model = fasttext.load_model('lid.176.bin')
    # sme_classifier(model, "text to classify")

if __name__ == '__main__':
    main()