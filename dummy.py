import tensorflow as tf

model = tf.keras.models.load_model('stock_price_prediction_model')

print(model.summary())

# import os
# print(os.path.exists('stock_price_prediction.keras'))  # Should return True
