
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = tf.keras.models.load_model('model')

with open('tokenizer.pickle', 'rb') as handle:
    tk = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

text='This is a sample text'
padded_text=pad_sequences(tk.texts_to_sequences([text]), maxlen=256, truncating='post')

prediction = model.predict(padded_text)
prediction = np.argmax(prediction, axis=-1)
print(prediction)

result = label_encoder.inverse_transform(prediction)
print(result[0])

# for item in prediction:
#     index=0
#     for ele in item:
#         print(labelencoder_Y.inverse_transform([index])+ " - " + str(round(ele*100,4)) + "%")
#         index+=1