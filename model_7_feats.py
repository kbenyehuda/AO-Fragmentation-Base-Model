
import tensorflow as tf
from tensorflow import keras 


################################################################################# 
#INPUT IS [224,224]
visible = tf.keras.layers.Input(shape=[224,224,1])
x=tf.keras.layers.Conv2D(64, kernel_size=4,padding='same',activation='relu',kernel_initializer='he_uniform',input_shape=[224,224,1])(visible)
x=tf.keras.layers.SpatialDropout2D(0.5)(x)
x=tf.keras.layers.MaxPool2D(pool_size=(2, 2),padding='same')(x)
x=tf.keras.layers.Conv2D(32,kernel_size=4,padding='same',activation='relu',kernel_initializer='he_uniform')(x)
x=tf.keras.layers.SpatialDropout2D(0.5)(x)
x=tf.keras.layers.MaxPool2D(pool_size=(2, 2),padding='same')(x)
x=tf.keras.layers.Conv2D(16,kernel_size=4,padding='same',activation='relu',kernel_initializer='he_uniform')(x)
x=tf.keras.layers.SpatialDropout2D(0.5)(x)
x=tf.keras.layers.MaxPool2D(pool_size=(2, 2),padding='same')(x)
x=tf.keras.layers.Conv2D(8, kernel_size=2,padding='same',activation='relu',kernel_initializer='he_uniform')(x)
x=tf.keras.layers.SpatialDropout2D(0.5)(x)
x=tf.keras.layers.MaxPool2D(pool_size=(2, 2),padding='same')(x)
x=tf.keras.layers.Conv2D(2, kernel_size=2,padding='same',activation='relu',kernel_initializer='he_uniform')(x)
x=tf.keras.layers.SpatialDropout2D(0.5)(x)
x=tf.keras.layers.MaxPool2D(pool_size=(2, 2),padding='same')(x)
x=tf.keras.layers.Flatten()(x)

model=tf.keras.Model(inputs=visible,outputs=x)

#model.summary()

##################################################################################
inp=tf.keras.Input(shape=(7,))

x=tf.keras.layers.Dense(14,kernel_initializer='he_uniform')(inp)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.Activation("relu")(x)
x=tf.keras.layers.Dropout(0.5)(x)
x=tf.keras.layers.Dense(49,kernel_initializer='he_uniform')(x)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.Activation("relu")(x)
x=tf.keras.layers.Dropout(0.5)(x)
x=tf.keras.layers.Dense(98,kernel_initializer='he_uniform')(x)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.Activation("relu")(x)


feat_model=tf.keras.Model(inputs=inp,outputs=x)


###################################################################################
optimizer = tf.optimizers.Adam(learning_rate=1e-2,beta_1=0.99,beta_2=0.999,amsgrad=True)

###################################################################################
def create_combined(model_1, model_2):

  # combine the output of the two branches
  
  #combined = tf.keras.layers.concatenate([model_1.output, model_2.output])
  combined = tf.keras.layers.concatenate([model_1.output,model_2.output])

  # apply a FC layer and then a regression prediction on the  
  # combined outputs
  #z_0 = tf.keras.layers.Dense(24, activation=keras.layers.LeakyReLU(alpha=0.1))(combined)
  z_0 = tf.keras.layers.Dense(24, activation=keras.layers.LeakyReLU(alpha=0.3),kernel_initializer='he_uniform')(combined)
  z_1 = tf.keras.layers.Dense(12, activation = keras.layers.LeakyReLU(alpha=0.3),kernel_initializer='he_uniform')(z_0)
  z_2 = tf.keras.layers.Dense(1, activation="sigmoid")(z_1)
 
  # our model will accept the inputs of the two branches and
  # then output a single value
  combined_model = tf.keras.Model(inputs=[model_1.input, model_2.input], outputs=z_2)

  #discriminator.trainable=False
  #gan_input = tf.keras.Input(shape=(100,))
  #x = generator(gan_input)
  #gan_output= discriminator(x)
  #gan= tf.keras.Model(inputs=gan_input, outputs=gan_output)
  combined_model.compile(loss='binary_crossentropy', optimizer='adam')
  #combined_model.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
  return combined_model

combined_model = create_combined(model,feat_model)
#combined_model.summary()




