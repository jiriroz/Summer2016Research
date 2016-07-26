from transfer import NeuralStyle

while True:
    """Below is just sample code. We need to read the specifications and images.
    Should I send the images or save them to disk and then read them?
    """
    message = raw_input()
    imgdata = base64.b64decode(message[22:]) #omit headers etc.
    frame = Image.open(BytesIO(imgdata))
    frame = np.array(frame)
    
