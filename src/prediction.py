
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = tf.keras.models.load_model('model')

with open('tokenizer.pickle', 'rb') as handle:
    tk = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

text="Induchoodan confronts his father and prods him to accept mistake and acknowledge the parentage of Indulekha. Menon ultimately regrets and secretly goes on to confess to his daughter. The very next morning when Induchoodan returns to Poovally, Indulekha is found dead and Menon is accused of murdering her. The whole act was planned by Pavithran, who after killing Indulekha, forces Raman Nair (Menon's longtime servant) to testify against Menon in court. In court, Nandagopal Marar, a close friend of Induchoodan and a supreme court lawyer, appears for Menon and manages to lay bare the murder plot and hidden intentions of other party. Menon is judged innocent of the crime by court"
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