import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:/Users/Ace/Desktop/Sem 3 Project/HTG-GAN-90feba63302a.json"
word_dict={}

def detect_document(path):
    
    wholepage=''
    """Detects document features in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            # print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                # print('Paragraph confidence: {}'.format(
                #     paragraph.confidence))
                wholepage+='\n'
                i=1
                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    # print('Word text: {} (confidence: {})'.format(
                    #     word_text, word.confidence))
                    wholepage+=f" {word_text}"
                    # word_dict.update({path+str(i) : word_text})
                    
                    # for symbol in word.symbols:
                    #     print('\tSymbol: {} (confidence: {})'.format(
                    #         symbol.text, symbol.confidence))
                    i+=1
    return wholepage
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

files= os.listdir('Total_Words/')
metalabel={}
for i in files: 
    word=detect_document('Total_Words/'+i)
    print(word)
    metalabel['i']=word
